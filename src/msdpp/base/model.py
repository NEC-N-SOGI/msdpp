from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from datasets import Image as DatasetsImage
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader

from msdpp.schema import RetrievalResults


class BaseModel(ABC):
    def __init__(
        self,
        pretrained_model: str | Path,
        n_text_batch: int = 100,
        *args: Any,  # noqa: ARG002,ANN401
        **kwargs: Any,  # noqa: ARG002,ANN401
    ) -> None:
        if isinstance(pretrained_model, Path):
            model_name = pretrained_model.name
        else:
            model_name = pretrained_model

        self.model_name = model_name.replace("/", "-")
        self.n_text_batch = n_text_batch

    @property
    def get_name(self) -> str:
        """Get the name of the diversification method."""
        return self.model_name

    def __str__(self) -> str:
        return self.get_name

    @abstractmethod
    def infer_image_text_similarity(
        self, image: Path | Image | Tensor, text: str
    ) -> Tensor:
        pass

    def prepare_loader(self, img_datasets: Dataset) -> DataLoader:
        if "image" not in img_datasets.column_names:
            msg = "image column is required."
            raise KeyError(msg)
        if not (isinstance(img_datasets[0]["image"], dict | Image)):
            msg = "image column should be dict (to be casted to DatasetsImage) or PIL.Image.Image"
            raise TypeError(msg)

        if isinstance(img_datasets[0]["image"], dict):
            img_datasets.cast_column("image", DatasetsImage())

        def collate_fn(batch: list[dict]) -> list[Image]:
            return [b["image"] for b in batch]

        dataset = img_datasets.with_format("torch")
        return DataLoader(
            dataset=dataset,  # pyright: ignore[reportArgumentType]
            batch_size=16,
            num_workers=8,
            collate_fn=collate_fn,
        )

    @abstractmethod
    def txt_img_sim(self, text: str | Tensor, img_feats: Tensor) -> Tensor:
        pass

    @abstractmethod
    def img_sim(self, src_feats: Tensor, dst_feats: Tensor | None) -> Tensor:
        pass

    @abstractmethod
    def infer_datasets(
        self, img_datasets: Dataset | DatasetDict, texts: list[str]
    ) -> RetrievalResults:
        pass
