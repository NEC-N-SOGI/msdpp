from dataclasses import dataclass

import torch

from msdpp.schema import EvalIndices, RetrievalResults


@dataclass
class TaskResult:
    org_indices: EvalIndices
    eval_indices: EvalIndices
    candidates: list[torch.Tensor]
    t2i_sim: torch.Tensor


@dataclass
class CacheResult:
    results: RetrievalResults
    retrieval_words: list[str]
