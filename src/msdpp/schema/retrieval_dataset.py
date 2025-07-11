from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict
from pydantic import StrictStr

from msdpp.schema.common import raise_error


@dataclass
class RetrievalDataset:
    name: StrictStr
    dataset: Dataset | DatasetDict
    retrieval_words: list[StrictStr]
    labels: dict[StrictStr, torch.Tensor]
    ext_data: torch.Tensor | list[torch.Tensor]

    def __len__(self) -> int:
        return len(self.dataset)

    def __post_init__(self) -> None:
        if isinstance(self.ext_data, list):
            for e in self.ext_data:
                raise_error(
                    e.shape[0] != len(self.dataset),
                    "ext_data must have the same length",
                )
        else:
            raise_error(
                len(self.dataset) != len(self.ext_data),
                "Dataset and ext_data must have the same length",
            )

        raise_error(
            len(self.retrieval_words) != len(self.labels),
            "retrieval_words and labels must have the same length",
        )

        raise_error(
            not all(word in self.labels for word in self.retrieval_words),
            "All retrieval_words must be in labels",
        )

        raise_error(
            any(
                labels.shape[0] != len(self.dataset)
                for labels in self.labels.values()
            ),
            "All labels must have the same length",
        )
