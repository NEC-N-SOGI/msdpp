from dataclasses import dataclass

import torch

from msdpp.schema.common import raise_error


@dataclass
class RetrievalResults:
    t2i_sim: torch.Tensor
    image_feats: torch.Tensor
    text_feats: torch.Tensor

    def __post_init__(self) -> None:
        raise_error(
            self.t2i_sim.shape[1] != self.image_feats.shape[0],
            f"({self.t2i_sim.shape[1]}, {self.image_feats.shape[0]})",
        )
        raise_error(
            self.t2i_sim.shape[0] != self.text_feats.shape[0],
            f"({self.t2i_sim.shape[0]}, {self.text_feats.shape[0]})",
        )

    def to(
        self, device: str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        if device is not None:
            self.t2i_sim = self.t2i_sim.to(device)
            self.image_feats = self.image_feats.to(device)
            self.text_feats = self.text_feats.to(device)

        if device is not None:
            self.t2i_sim = self.t2i_sim.to(dtype)
            self.image_feats = self.image_feats.to(dtype)
            self.text_feats = self.text_feats.to(dtype)
