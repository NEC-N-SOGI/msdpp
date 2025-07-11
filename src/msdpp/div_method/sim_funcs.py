import torch

from msdpp import registry


def _normalize_sim(sim: torch.Tensor) -> torch.Tensor:
    sim -= sim.diag().diag()
    sim = sim / sim.max()
    sim += torch.eye(sim.shape[0], device=sim.device)
    return sim


@registry.register_sim_func("inner_product")
def inner_product(data: torch.Tensor, do_normalize: bool = True) -> torch.Tensor:
    ip = data @ data.T

    if do_normalize:
        ip = _normalize_sim(ip)

    return ip


@registry.register_sim_func("dist_inv")
def dist_inv(data: torch.Tensor, do_normalize: bool = True) -> torch.Tensor:
    dist: torch.Tensor = 1 / (torch.cdist(data, data) + 1)

    if do_normalize:
        dist = _normalize_sim(dist)

    return dist


@registry.register_sim_func("rbf")
def rbf(data: torch.Tensor, do_normalize: bool = True) -> torch.Tensor:
    dist = torch.cdist(data, data)
    tril = dist[torch.tril(torch.ones_like(dist), -1).bool()]

    rbf_val = (-torch.cdist(data, data) / tril.median() / 2).exp()

    if do_normalize:
        rbf_val = _normalize_sim(rbf_val)

    return rbf_val
