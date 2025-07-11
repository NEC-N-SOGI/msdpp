import numpy as np
import torch

from msdpp import registry
from msdpp.base.divmethod import DivDir
from msdpp.schema import EvalIndices

MAX_DATA_DIM = 2


class EvalIndexCalculator:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def calc_map_at_k(
        self, pred_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = self.top_k
        if pred_label.ndim == 1:
            pred_label = pred_label.unsqueeze(0)

        pred_at_k = pred_label[:, : self.top_k]
        ap = (pred_at_k.cumsum(1) / (torch.arange(top_k) + 1) * pred_at_k).sum(
            1
        ) / pred_at_k.sum(1)
        ap[ap.isnan()] = 0

        mean_ap = ap.mean()

        return ap, mean_ap

    def calc_r_at_k(
        self,
        pred_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred_label.ndim == 1:
            pred_label = pred_label.unsqueeze(0)

        pred_label = pred_label.to(torch.float32)
        r1 = pred_label[:, :1].max(-1)[0].mean()
        r5 = pred_label[:, :5].max(-1)[0].mean()
        r10 = pred_label[:, :10].max(-1)[0].mean()

        return r1, r5, r10

    def calc_mrr(self, pred_label: torch.Tensor) -> torch.Tensor:
        if pred_label.ndim == 1:
            pred_label = pred_label.unsqueeze(0)

        pred_label = pred_label.to(torch.float32)
        mrr: torch.Tensor = (1 / (pred_label.argmax(-1) + 1)).mean()

        return mrr

    def calc_ilad_ilmd(
        self, feats: torch.Tensor, idxs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.top_k
        ilad = torch.zeros(idxs.shape[0])
        ilmd = torch.zeros(idxs.shape[0])

        _feats = feats.cuda().float()

        for i, idx in enumerate(idxs):
            dist_matrix = torch.cdist(
                _feats[idx[:k]].float(), _feats[idx[:k]].float()
            )

            dist_sum = dist_matrix.sum() - dist_matrix.diag().sum()
            ilad[i] = dist_sum.cpu() / (k * (k - 1))
            ilmd[i] = (dist_matrix.cpu() + torch.eye(k) * 1e6).min()

        return ilad, ilmd

    def calc_vendi_score(
        self, ext_data: torch.Tensor | list[torch.Tensor], pred_idxs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.top_k
        if not isinstance(ext_data, list):
            ext_data = [ext_data]

        sim_func = registry.mapping.sim_funcs["dist_inv"]
        eye = 1e-5 * torch.eye(k).cuda()

        sim_mats = torch.zeros((len(ext_data), len(pred_idxs), k, k)).cuda()
        for i, data_pt in enumerate(ext_data):  # each attribute
            _data_pt = data_pt.cuda()
            if data_pt.ndim > MAX_DATA_DIM:
                _data_pt = _data_pt.mean(1)
            for j, idx in enumerate(pred_idxs):  # each query
                sim_mat = sim_func(_data_pt[idx[:k]].float(), False)
                sim_mat /= k
                sim_mats[i, j] = sim_mat + eye

        # vs_q: https://arxiv.org/abs/2310.12952
        q = 0.1
        # 1~k
        eigvals: torch.Tensor = torch.linalg.eigvalsh(sim_mats.cuda())
        vendi_score = (eigvals.pow(q).sum(-1).log() / (1 - q)).exp()
        # 0~1
        vendi_score_norm = (vendi_score.cpu() - 1) / (k - 1)

        ext_mean_score = vendi_score_norm.mean(0)

        return ext_mean_score.mean(), vendi_score_norm

    def calc_ncs(self, spice_mat_path: str, pred_idxs: torch.Tensor) -> torch.Tensor:
        spice_mat = torch.load(spice_mat_path)
        if isinstance(spice_mat, np.ndarray):
            spice_mat = torch.from_numpy(spice_mat)
        spice_mat = spice_mat.cuda()
        ncs = torch.zeros(pred_idxs.shape[0])
        sorted_spice = spice_mat.sort(dim=0, descending=True).values[:10, :]
        for i, pred in enumerate(pred_idxs):
            ncs[i] = spice_mat[pred[:10], i].sum() / sorted_spice[:, i].sum()

        return ncs.mean().cpu()

    def run(
        self,
        t2i_sim: torch.Tensor,
        labels: list,
        feats: torch.Tensor,
        ext_data: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        spice_mat_path: str = "",
    ) -> EvalIndices:
        _, idxs = t2i_sim.sort(dim=-1, descending=True)

        sorted_labels = torch.gather(torch.stack(labels), 1, idxs)

        if feats.ndim > MAX_DATA_DIM:
            feats = feats.cuda().mean(1)
            feats = torch.nn.functional.normalize(feats, dim=-1)

        feats = feats.cpu()

        if isinstance(ext_data, list):
            ext_data = torch.cat(ext_data, dim=1)

        ext_data = ext_data.cpu()
        sorted_labels = sorted_labels.cpu()
        idxs = idxs.cpu()

        ap, map_ = self.calc_map_at_k(sorted_labels)
        img_ilad, img_ilmd = self.calc_ilad_ilmd(feats, idxs)
        ext_ilad, ext_ilmd = self.calc_ilad_ilmd(ext_data, idxs)

        r1, r5, r10 = self.calc_r_at_k(sorted_labels)
        mrr = self.calc_mrr(sorted_labels)

        ncs = torch.tensor(1.0)
        if spice_mat_path != "":
            ncs = self.calc_ncs(spice_mat_path, idxs)

        atts = [ext_data] if not isinstance(ext_data, list) else ext_data  # type: ignore[unreachable]
        atts.append(feats)

        m_img_vendi_score, img_vendi_scores = self.calc_vendi_score(feats, idxs)
        m_ext_vendi_score, ext_vendi_scores = self.calc_vendi_score(ext_data, idxs)

        if direction == DivDir.DECREASE:
            m_ext_vendi_score = 1 - m_ext_vendi_score
            ext_vendi_scores = 1 - ext_vendi_scores

        all_vendi_scores = torch.cat([img_vendi_scores, ext_vendi_scores], dim=0)
        m_vendi_per_retrieval = 1 / (1 / all_vendi_scores).mean(0)
        m_all_vendi_score = 1 / (1 / m_vendi_per_retrieval).mean()

        return EvalIndices(
            ap=ap,
            map_=map_,
            r1=r1,
            r5=r5,
            r10=r10,
            mrr=mrr,
            ncs=ncs,
            img_ilad=img_ilad,
            img_ilmd=img_ilmd,
            ext_ilad=ext_ilad,
            ext_ilmd=ext_ilmd,
            img_vendi=m_img_vendi_score,
            ext_vendi=m_ext_vendi_score,
            mean_vendi=m_all_vendi_score,
            img_vendi_scores=img_vendi_scores,
            ext_vendi_scores=ext_vendi_scores,
            all_vendi_scores=all_vendi_scores,
        )
