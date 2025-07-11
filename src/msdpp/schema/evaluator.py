from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EvalIndices:
    ap: torch.Tensor
    map_: torch.Tensor
    r1: torch.Tensor
    r5: torch.Tensor
    r10: torch.Tensor
    mrr: torch.Tensor
    ncs: torch.Tensor
    img_ilad: torch.Tensor
    img_ilmd: torch.Tensor
    ext_ilad: torch.Tensor
    ext_ilmd: torch.Tensor
    img_vendi: torch.Tensor
    ext_vendi: torch.Tensor
    mean_vendi: torch.Tensor
    img_vendi_scores: torch.Tensor
    ext_vendi_scores: torch.Tensor
    all_vendi_scores: torch.Tensor

    def harmonic_mean(self, data: torch.Tensor) -> torch.Tensor:
        n = data.shape[0]
        hm: torch.Tensor = n / (1 / data).sum()
        return hm

    def calc_all_hm(self) -> dict[str, torch.Tensor]:
        mean_combinations = {
            "m_map_img_ilad": [self.map_, self.m_img_ilad],  # map + 1 div
            "m_map_img_ilmd": [self.map_, self.m_img_ilmd],
            "m_map_vendi": [self.map_, self.mean_vendi],
            "m_r1_img_ilad": [self.r1, self.m_img_ilad],  # r1 + 1 div
            "m_r1_img_ilmd": [self.r1, self.m_img_ilmd],
            "m_r1_vendi": [self.r1, self.mean_vendi],
            "m_r5_img_ilad": [self.r5, self.m_img_ilad],  # r5 + 1 div
            "m_r5_img_ilmd": [self.r5, self.m_img_ilmd],
            "m_r5_vendi": [self.r5, self.mean_vendi],
            "m_r10_img_ilad": [self.r10, self.m_img_ilad],  # r10 + 1 div
            "m_r10_img_ilmd": [self.r10, self.m_img_ilmd],
            "m_r10_vendi": [self.r10, self.mean_vendi],
            "m_mrr_img_ilad": [self.mrr, self.m_img_ilad],  # mrr + 1 div
            "m_mrr_img_ilmd": [self.mrr, self.m_img_ilmd],
            "m_mrr_vendi": [self.mrr, self.mean_vendi],
            "m_ncs_img_ilad": [self.ncs, self.m_img_ilad],  # ncs + 1 div
            "m_ncs_img_ilmd": [self.ncs, self.m_img_ilmd],
            "m_ncs_vendi": [self.ncs, self.mean_vendi],
        }
        harmonic_means = {}

        for key, values in mean_combinations.items():
            harmonic_means[key] = self.harmonic_mean(torch.stack(values))

        return harmonic_means

    # post process
    def __post_init__(self) -> None:
        self.m_img_ilad = self.img_ilad.mean()
        self.m_img_ilmd = self.img_ilmd.mean()
        self.m_ext_ilad = self.ext_ilad.mean()
        self.m_ext_ilmd = self.ext_ilmd.mean()

        self.harmonic_means = {"hm": self.calc_all_hm()}

    def to_table(self) -> dict[str, list[str] | str]:
        dict_data = {
            "AP": self.ap,
            "MAP": self.map_.item(),
            "R_1": self.r1.item(),
            "R_5": self.r5.item(),
            "R_10": self.r10.item(),
            "MRR": self.mrr,
            "m Image ILAD": self.m_img_ilad.item(),
            "m Image ILMD": self.m_img_ilmd.item(),
            "m Ext ILAD": self.m_ext_ilad.item(),
            "m Ext ILMD": self.m_ext_ilmd.item(),
            "Image ILAD": self.img_ilad,
            "Image ILMD": self.img_ilmd,
            "Ext ILAD": self.ext_ilad,
            "Ext ILMD": self.ext_ilmd,
        }

        table: dict[str, str | list[str]] = {}

        for key, value in dict_data.items():
            if isinstance(value, torch.Tensor):
                value_list = value.cpu().tolist()
                if isinstance(value_list, list):
                    table[key] = [f"{v:.4f}" for v in value_list]
                else:
                    table[key] = f"{value:.4f}"  # type: ignore[unreachable]
            else:
                table[key] = f"{value:.4f}"

        return table

    def to_dict(self) -> dict[str, dict[str, list[str] | str]]:
        # unify all values including harmonic_means to list
        dict_data = {
            "AP": self.ap.tolist(),
            "MAP": self.map_.item(),
            "R_1": self.r1.item(),
            "R_5": self.r5.item(),
            "R_10": self.r10.item(),
            "MRR": self.mrr.item(),
            "img_vendi": self.img_vendi.item(),
            "ext_vendi": self.ext_vendi.item(),
            "mean_vendi": self.mean_vendi.item(),
            "m Image ILAD": self.m_img_ilad.item(),
            "m Image ILMD": self.m_img_ilmd.item(),
            "m Ext ILAD": self.m_ext_ilad.item(),
            "m Ext ILMD": self.m_ext_ilmd.item(),
            "Image ILAD": self.img_ilad.tolist(),
            "Image ILMD": self.img_ilmd.tolist(),
            "Ext ILAD": self.ext_ilad.tolist(),
            "Ext ILMD": self.ext_ilmd.tolist(),
        }

        dicts: dict[str, dict[str, Any]] = {}

        for key, hms in self.harmonic_means.items():
            dicts[key] = dict_data | {_k: _v.tolist() for _k, _v in hms.items()}

            # values to str with format .4f
            for k, value in dicts[key].items():
                if isinstance(value, list):
                    dicts[key][k] = [f"{v:.4f}" for v in value]
                else:
                    dicts[key][k] = f"{value:.4f}"

        return dicts
