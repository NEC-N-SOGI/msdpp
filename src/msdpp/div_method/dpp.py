import torch
from pydantic import BaseModel, Field, StrictFloat, StrictStr, field_validator

from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod, DivDir

BETA_MIN = 0.01
BETA_MAX = 0.99


class DPPParams(BaseModel):
    theta: StrictFloat = Field(1.0, ge=0, le=1)
    beta: StrictFloat = Field(0.5, ge=0, le=1)
    sim_func_name: StrictStr = Field("dist_inv", alias="sim_func_name")

    @field_validator("sim_func_name")
    @classmethod
    def check_sim_func_name(cls, v: StrictStr) -> StrictStr:
        if v not in registry.mapping.sim_funcs:
            msg = f"{v} is not in {registry.mapping.sim_funcs.keys()}"
            raise KeyError(msg)

        return v


@registry.register_div_method("dpp_sim_average")
class DPPGreedy(BaseDiversificationMethod):
    def __init__(
        self,
        theta: float = 1.0,
        beta: float = 0.5,
        sim_func_name: str = "dist_inv",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.params = DPPParams(theta=theta, beta=beta, sim_func_name=sim_func_name)
        self.beta = beta
        self.theta = theta

        self.sim_func = registry.mapping.sim_funcs[sim_func_name]
        self.device = device

    def _normalize_sim(self, sim: torch.Tensor) -> torch.Tensor:
        sim -= sim.diag().diag()
        sim += torch.eye(sim.shape[0], device=sim.device)
        return sim

    def ext_kernel_matrix(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        data_list = [data] if isinstance(data, torch.Tensor) else data

        k_mat = torch.zeros(
            (data_list[0].shape[0], data_list[0].shape[0]),
            device=self.device,
        )

        for data_item in data_list:
            cuda_data = data_item.to(self.device)
            k_mat += self.sim_func(cuda_data, False)

        k_mat /= len(data_list)
        return k_mat.float()

    def img_kernel_matrix(self, img_feats: torch.Tensor) -> torch.Tensor:
        if img_feats.dim() == 1:
            img_feats = img_feats.unsqueeze(0)

        if img_feats.dim() == 3:  # noqa: PLR2004
            img_feats = img_feats.mean(1)

        norm_img_feats = torch.nn.functional.normalize(
            img_feats.to(self.device), dim=-1
        )

        k_mat: torch.Tensor = self.sim_func(norm_img_feats, False)

        return k_mat

    def calc_alpha(self, r: torch.Tensor) -> torch.Tensor:
        return (self.theta / (2 * (1 - self.theta)) * r).exp()

    def calc_similarity_matrix(
        self,
        info: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        r: torch.Tensor,
        img_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # r is needed for subclass

        _ext_sim = self.ext_kernel_matrix(info)  # gpu
        sign = 1 if direction == DivDir.INCREASE else -1
        ext_sim = sign * _ext_sim

        if img_feats is not None and self.beta > BETA_MIN:
            img_sim = self.img_kernel_matrix(img_feats)  # gpu

            if self.beta < BETA_MAX:
                sim = (1 - self.beta) * ext_sim + self.beta * img_sim
            else:
                sim = img_sim
        else:
            sim = ext_sim

        return sim.to(torch.float32), r.to(sim.device)

    def search(
        self,
        info: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        t2i_sim: torch.Tensor | None = None,
        top_k: int = 100,
        img_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_data = info[0].shape[0] if isinstance(info, list) else info.shape[0]

        r = t2i_sim.squeeze() if t2i_sim is not None else torch.ones(n_data)
        r = r.to(torch.float32)

        sim, r = self.calc_similarity_matrix(
            info=info, direction=direction, img_feats=img_feats, r=r
        )

        eps = 1e-10

        results = []

        weighted_r = self.calc_alpha(r).diag()
        l_matrix = weighted_r @ sim @ weighted_r

        # Implementation of Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
        # arXiv: https://arxiv.org/abs/1709.05135
        all_set = set(range(n_data))
        c = torch.zeros((n_data, top_k), device=sim.device)
        d = l_matrix.diag()
        j = int(d.argmax().item())
        yg = [j]

        for iter_i in range(top_k - 1):
            i_list = list(all_set - set(yg))

            cji = c[j, :iter_i]
            if cji.dim() == 2:  # noqa: PLR2004
                cji = cji.permute((0, 1))
            cij = c[i_list, :iter_i] @ cji

            ei = (l_matrix[j, i_list] - cij) / (d[j] + eps)

            c[i_list, iter_i] = ei
            d[i_list] -= ei**2

            d[yg] = 0
            j = int(d.argmax().item())
            if d[j] <= 0:  # if the variance is negative, break
                # append remaining number of indices to yg, with t2i_sim descending order
                _, idx = r.sort(dim=0, descending=True)
                for i in idx:
                    if i not in yg:
                        yg.append(int(i.item()))
                        if len(yg) == top_k:
                            break
                break
            yg.append(j)

        results.append(torch.tensor(yg).cpu())

        return torch.stack(results).cpu()

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPGreedy"


@registry.register_div_method("msdpp")
class MSDPP(DPPGreedy):
    def __init__(
        self,
        theta: float = 1.0,
        beta: float = 0.5,
        sim_func_name: str = "dist_inv",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.params = DPPParams(
            theta=theta,
            beta=beta,
            sim_func_name=sim_func_name,
        )
        self.beta = beta
        self.theta = theta
        self.norm_factor = "one"

        self.sim_func = registry.mapping.sim_funcs[sim_func_name]
        self.device = device

    def _logm_pd(self, mat: torch.Tensor) -> torch.Tensor:
        eye = 1e-3 * torch.eye(mat.shape[-1], device=mat.device)

        eig_val, eig_vec = torch.linalg.eigh(mat + eye)
        # for numerical stability; PD mat should have only non-negative eigenvalues
        eig_val[eig_val < 0] = 1

        log_mat: torch.Tensor = (
            eig_vec @ torch.diag_embed(eig_val.log()) @ eig_vec.transpose(-2, -1)
        )
        return log_mat

    def _expm(self, mat: torch.Tensor) -> torch.Tensor:
        eig_val, eig_vec = torch.linalg.eigh(mat)
        eig_val_exp = eig_val.exp()
        exp_mat: torch.Tensor = (
            eig_vec @ torch.diag_embed(eig_val_exp) @ eig_vec.transpose(-2, -1)
        )
        return exp_mat

    def ext_kernel_matrix(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        data_list = [data] if isinstance(data, torch.Tensor) else data

        k_mat = []
        for data_item in data_list:
            cuda_data = data_item.to(self.device)
            k_mat.append(self.sim_func(cuda_data, False))

        return torch.stack(k_mat).float()

    def _log_normalize(
        self,
        sim: torch.Tensor,
        ext_log: torch.Tensor,
        img_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sim, ext_log, img_log, ext_log.mean(0)

    def mean_normalize(
        self,
        mean_log: torch.Tensor,
        r: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return mean_log

    def calc_similarity_matrix(
        self,
        info: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        r: torch.Tensor,
        img_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext_sim = self.ext_kernel_matrix(info)  # gpu

        # if img_feats is not None and self.beta > 0.01:
        if img_feats is None:
            msg = "img_feats must be provided"
            raise ValueError(msg)

        img_sim = self.img_kernel_matrix(img_feats)  # gpu

        ext_log = self._logm_pd(ext_sim)
        img_log = self._logm_pd(img_sim)

        sign = 1 if direction == DivDir.INCREASE else -1

        r, _, img_log, m_ext_log = self._log_normalize(r, ext_log, img_log)

        # beta=0 --> sim = inv(ext_sim) --> localize ext data
        mean_log = sign * (1 - self.beta) * m_ext_log + self.beta * img_log
        mean_log_norm = self.mean_normalize(mean_log, r)
        sim = self._expm(mean_log_norm)

        return sim.to(torch.float32), r.to(sim.device)

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "MSDPP"


@registry.register_div_method("msdpp_tn")
class MSDPPTanNorm(MSDPP):
    def _calc_factor(
        self, ext_log: torch.Tensor, img_log: torch.Tensor, sim: torch.Tensor
    ) -> torch.Tensor:
        if self.norm_factor == "max":
            max_factor: torch.Tensor = torch.cat(
                [
                    ext_log.view(ext_log.shape[0], -1).norm(2, -1),
                    img_log.norm().unsqueeze(0),
                ],
                0,
            ).max()
            return max_factor

        if self.norm_factor == "mean":
            mean_factor: torch.Tensor = torch.cat(
                [
                    ext_log.view(ext_log.shape[0], -1).norm(2, -1),
                    img_log.norm().unsqueeze(0),
                ],
                0,
            ).mean()
            return mean_factor

        sim_log = sim.log()
        sim_log_norm = sim_log.norm().squeeze()
        factor: torch.Tensor = sim_log_norm.to(img_log.device)
        return factor

    def _log_normalize(
        self,
        sim: torch.Tensor,
        ext_log: torch.Tensor,
        img_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """.

        Args:
            sim (torch.Tensor): 2d tensor
            ext_log (torch.Tensor): 3d tensor
            img_log (torch.Tensor): 2d tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        factor = self._calc_factor(ext_log, img_log, sim)

        ext_log_norm = (
            factor
            * ext_log
            / ext_log.view(ext_log.shape[0], -1)
            .norm(2, -1, keepdim=True)
            .unsqueeze(2)
        )
        img_log_norm = factor * img_log / img_log.norm()

        m_ext_log = ext_log_norm.mean(0)
        m_ext_log_norm = factor * m_ext_log / m_ext_log.norm()
        return sim, ext_log_norm, img_log_norm, m_ext_log_norm

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "MSDPPTanNorm"


@registry.register_div_method("msdpp_mean_norm")
class MSDPPMeanNorm(MSDPP):
    def mean_normalize(
        self, mean_log: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        r_log = r.log()
        r_log_norm = r_log.norm()
        normalized_mean: torch.Tensor = r_log_norm * mean_log / mean_log.norm()
        return normalized_mean

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPMeanNorm"


@registry.register_div_method("msdpp_score_norm")
class MSDPPScoreNorm(MSDPP):
    def _log_normalize(
        self,
        sim: torch.Tensor,
        ext_log: torch.Tensor,
        img_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """.

        Args:
            sim (torch.Tensor): 2d tensor
            ext_log (torch.Tensor): 3d tensor
            img_log (torch.Tensor): 2d tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        sim_log = sim.log()
        sim_log_norm = sim_log / sim_log.norm()
        sim_log_norm_exp = sim_log_norm.exp()
        return sim_log_norm_exp, ext_log, img_log, ext_log.mean(0)

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPScoreNorm"


@registry.register_div_method("msdpp_mean_tan_norm")
class MSDPPMeanTanNorm(MSDPPTanNorm, MSDPPMeanNorm):
    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPMeanTanNorm"


@registry.register_div_method("msdpp_score_tan_norm")
class MSDPPScoreTanNorm(MSDPP):
    def _log_normalize(
        self,
        sim: torch.Tensor,
        ext_log: torch.Tensor,
        img_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """.

        Args:
            sim (torch.Tensor): 2d tensor
            ext_log (torch.Tensor): 3d tensor
            img_log (torch.Tensor): 2d tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        ext_log_norm = ext_log / ext_log.view(ext_log.shape[0], -1).norm(
            2, -1, keepdim=True
        ).unsqueeze(2)
        img_log_norm = img_log / img_log.norm()
        sim_log = sim.log()
        sim_log_norm = sim_log / sim_log.norm()
        sim_log_norm_exp = sim_log_norm.exp()

        return sim_log_norm_exp, ext_log_norm, img_log_norm, ext_log_norm.mean(0)

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPScoreTanNorm"


@registry.register_div_method("msdpp_score_mean_norm")
class MSDPPScoreMeanNorm(MSDPPScoreNorm, MSDPPMeanNorm):
    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPScoreMeanNorm"


@registry.register_div_method("msdpp_tn_tvms")
class MSDPPScoreMeanTanNorm(MSDPPScoreTanNorm, MSDPPMeanNorm):
    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return "DPPScoreMeanTanNorm"
