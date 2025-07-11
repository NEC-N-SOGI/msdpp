from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pydantic import BaseModel


class DivDir(Enum):
    INCREASE = auto()
    DECREASE = auto()


class BaseDiversificationMethod(ABC):
    """Base class for diversification methods."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.params: BaseModel

    @abstractmethod
    def search(
        self,
        info: torch.Tensor | list[torch.Tensor],
        direction: DivDir,
        t2i_sim: torch.Tensor | None = None,
        top_k: int = 100000,
        img_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    def diversify(
        self,
        t2i_sim: torch.Tensor,
        target_info: torch.Tensor | list[torch.Tensor],
        direction: DivDir = DivDir.INCREASE,
        top_k: int = 100000,
        img_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(target_info, list):
            for i in range(len(target_info)):
                if target_info[i].dim() == 1:
                    target_info[i] = target_info[i].unsqueeze(1)
        elif target_info.dim() == 1:
            target_info = target_info.unsqueeze(1)
        # search orders of indexes to maximize/minimize AUC
        index = self.search(
            target_info, direction, t2i_sim=t2i_sim, top_k=top_k, img_feats=img_feats
        )

        return index.squeeze()

    @property
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the diversification method."""

    def __str__(self) -> str:
        return self.get_name

    @property
    def get_params(self) -> dict:
        """Get the parameters of the diversification method."""
        return self.params.model_dump()
