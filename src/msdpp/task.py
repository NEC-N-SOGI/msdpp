import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from msdpp.base.divmethod import (
    BaseDiversificationMethod,
    DivDir,
)
from msdpp.base.model import BaseModel, RetrievalResults
from msdpp.evalindex.eval_index import EvalIndexCalculator
from msdpp.schema import (
    CacheResult,
    RetrievalDataset,
    TaskResult,
)

CACHE_HOME = Path(os.environ.get("HF_HOME", "./")) / "div_results"


class BaseTask:
    def __init__(
        self,
        model: BaseModel,
        dataset: RetrievalDataset,
        subset_k: int = 200,
        top_k: int = 50,
        cache_home: Path = CACHE_HOME,
        do_cache_sim: bool = True,
        n_thread: int = 1,
    ) -> None:
        self.model = model
        self.subset_k = subset_k
        self.top_k = top_k
        self.cache_home = cache_home
        self.do_cache_sim = do_cache_sim
        self.n_thread = n_thread

        self.eval_calculator = EvalIndexCalculator(top_k)

        # if retrieval results are cached, load them
        self.retrieval_results = self.load_cache_sim(
            str(self.model), dataset.name, dataset.retrieval_words
        )
        self.dataset = dataset

        if not cache_home.exists():
            cache_home.mkdir(exist_ok=True, parents=True)

    def _cache_sim_path(self, model_name: str, data_name: str) -> Path:
        # generate a cache path of retrieval results
        # based on model name and dataset name
        _data_name = data_name.split("_")[0].lower()

        if data_name.endswith("val"):
            _data_name += "_val"

        if data_name.endswith("test"):
            _data_name += "_test"

        return self.cache_home / f"{model_name!s}_{_data_name}_ret_results.pkl"

    def load_cache_sim(
        self, model_name: str, data_name: str, retrieval_words: list[str]
    ) -> RetrievalResults | None:
        # load cached retrieval results
        # if retrieval words are changed or no cache file exists,
        # return None to re-run the retrieval process
        result_path = self._cache_sim_path(model_name, data_name)

        if not Path(result_path).exists():  # No cache file
            return None

        cached_sim: CacheResult = torch.load(result_path, weights_only=False)
        match_words = all(i in cached_sim.retrieval_words for i in retrieval_words)

        if not match_words:  # retrieval words are changed
            return None

        # consistent indices of cached words with retrieval words
        idxs = [cached_sim.retrieval_words.index(word) for word in retrieval_words]

        cached_sim.results.t2i_sim = cached_sim.results.t2i_sim[idxs].contiguous()
        cached_sim.results.text_feats = cached_sim.results.text_feats[
            idxs
        ].contiguous()

        return cached_sim.results

    def cache_sim(
        self,
        model_name: str,
        data_name: str,
        results: RetrievalResults,
        retrieval_words: list[str],
    ) -> None:
        # cache retrieval results
        t2i_sim_path = self._cache_sim_path(model_name, data_name)
        results.to("cpu")

        torch.save(
            CacheResult(results=results, retrieval_words=retrieval_words),
            t2i_sim_path,
        )

    def _cache_rerank_path(
        self,
        model_name: str,
        data_name: str,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
    ) -> tuple[Path, Path]:
        # generate a cache path of rerank or CDR-CA task results
        # based on model name, dataset name, diversification method, and direction
        params = div_method.get_params
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])

        data_name_split = data_name.split("_")
        if "pp_" in data_name.lower():
            _data_name = "_".join(
                [data_name_split[0].lower(), data_name_split[1].lower()]
            )
        else:
            _data_name = data_name.split("_")[0].lower()
        if data_name.endswith("val"):
            _data_name += "_val"
        if data_name.endswith("test"):
            _data_name += "_test"

        div_nmethod_name = div_method.get_name

        rerank_dir = self.cache_home / "rerank"
        if not rerank_dir.exists():
            rerank_dir.mkdir(exist_ok=True)

        direction_name = "increase" if direction == DivDir.INCREASE else "decrease"

        return (
            rerank_dir
            / f"{model_name!s}_{_data_name}_{direction_name}_{div_nmethod_name}_{param_str}.pkl"
        ), (
            rerank_dir
            / f"index_{model_name!s}_{_data_name}_{direction_name}_{div_nmethod_name}_{param_str}.pkl"
        )

    def load_cache_rerank(
        self,
        model_name: str,
        data_name: str,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
    ) -> list[torch.Tensor] | None:
        # load cached rerank or CDR-CA task results
        # if no cache file exists, return None to re-run the task
        result_path, _ = self._cache_rerank_path(
            model_name, data_name, div_method, direction
        )

        if not Path(result_path).exists():
            return None

        results: list[torch.Tensor] = torch.load(result_path, weights_only=False)
        return results

    def cache_rerank(
        self,
        result: TaskResult,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
    ) -> None:
        # cache rerank or CDR-CA task results
        # based on model name, dataset name, diversification method, and direction
        result_path, index_path = self._cache_rerank_path(
            str(self.model), self.dataset.name, div_method, direction
        )

        candidates: list[torch.Tensor] = result.candidates
        torch.save(candidates, result_path)

        torch.save(
            {"org": result.org_indices, "eval": result.eval_indices}, index_path
        )

    def _retrieve(self) -> RetrievalResults:
        # run retrieval process
        # if retrieval results are cached, return them
        # otherwise, run the model inference and cache the results
        dataset = self.dataset
        retrieval_words = dataset.retrieval_words
        retrieval_results = self.retrieval_results

        if retrieval_results is None:
            retrieval_results = self.model.infer_datasets(
                dataset.dataset,
                texts=retrieval_words,
            )
            if self.do_cache_sim:
                self.cache_sim(
                    str(self.model), dataset.name, retrieval_results, retrieval_words
                )
                self.retrieval_results = retrieval_results

        return retrieval_results

    def _run_trial(
        self,
        i: int,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
        t2i_sim: torch.Tensor,
        image_feats: torch.Tensor,
        ext_data: torch.Tensor | list[torch.Tensor],
        subset_k: int,
        top_k: int,
    ) -> tuple[int, torch.Tensor]:
        # run a single trial of diversification
        # for the i-th retrieval word
        topk_idx = t2i_sim[i].argsort(descending=True).cpu()[:subset_k]
        topk_t2i_sim = t2i_sim[i][topk_idx]

        topk_ext_data: torch.Tensor | list[torch.Tensor]
        if isinstance(ext_data, list):
            topk_ext_data = [e[topk_idx] for e in ext_data]
        else:
            topk_ext_data = ext_data[topk_idx]

        # diversify-localilze the top-k subset
        candidates = div_method.diversify(
            topk_t2i_sim,
            topk_ext_data,
            direction,
            top_k,
            image_feats[topk_idx],
        )
        # re-indexing to original index
        return i, torch.stack([topk_idx[j] for j in candidates])

    def _run(
        self,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
        t2i_sim: torch.Tensor,
        image_feats: torch.Tensor,
        ext_data: torch.Tensor | list[torch.Tensor],
        retrieval_words: list[str],
        subset_k: int,
        top_k: int,
    ) -> list[torch.Tensor]:
        # run diversification trials for all retrieval words
        all_candidates = []
        with torch.inference_mode():
            for i in range(len(retrieval_words)):
                _, indices = self._run_trial(
                    i,
                    div_method,
                    direction,
                    t2i_sim,
                    image_feats,
                    ext_data,
                    subset_k,
                    top_k,
                )
                all_candidates.append(indices)

        return all_candidates

    def _run_parallel(
        self,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
        t2i_sim: torch.Tensor,
        image_feats: torch.Tensor,
        ext_data: torch.Tensor | list[torch.Tensor],
        retrieval_words: list[str],
        subset_k: int,
        top_k: int,
    ) -> list[torch.Tensor]:
        # run diversification trials for all retrieval words in parallel
        with torch.inference_mode(), ThreadPoolExecutor(self.n_thread) as executor:
            unsorted_candidates = list(
                executor.map(
                    lambda i: self._run_trial(
                        i,
                        div_method,
                        direction,
                        t2i_sim,
                        image_feats,
                        ext_data,
                        subset_k,
                        top_k,
                    ),
                    range(len(retrieval_words)),
                )
            )

            # sort by original index
            sorted_candidates = sorted(unsorted_candidates, key=lambda x: x[0])
            return [x[1] for x in sorted_candidates]

    def run(
        self,
        div_method: BaseDiversificationMethod,
        direction: DivDir,
        force: bool = False,
        spice_mat_path: str = "",
    ) -> TaskResult:
        dataset = self.dataset
        retrieval_words = dataset.retrieval_words
        labels = dataset.labels
        labels_list = list(labels.values())
        ext_data = dataset.ext_data

        subset_k = self.subset_k
        top_k = self.top_k

        # Original retrieval results
        retrieval_results = self._retrieve()
        t2i_sim = retrieval_results.t2i_sim
        t2i_sim = t2i_sim.to("cpu")

        original_results = self.eval_calculator.run(
            t2i_sim,
            labels_list,
            retrieval_results.image_feats,
            ext_data,
            direction,
            spice_mat_path,
        )

        all_candidates = []

        cached_result = None
        if not force:
            # load cached rerank or CDR-CA task results
            cached_result = self.load_cache_rerank(
                str(self.model), self.dataset.name, div_method, direction
            )
            if cached_result is not None:
                all_candidates = cached_result

        if len(all_candidates) == 0:
            # if no cached results, run the task
            if self.n_thread <= 1:
                all_candidates = self._run(
                    div_method,
                    direction,
                    t2i_sim,
                    retrieval_results.image_feats,
                    ext_data,
                    retrieval_words,
                    subset_k,
                    top_k,
                )
            else:
                all_candidates = self._run_parallel(
                    div_method,
                    direction,
                    t2i_sim,
                    retrieval_results.image_feats,
                    ext_data,
                    retrieval_words,
                    subset_k,
                    top_k,
                )
        best_idxs = torch.zeros((len(retrieval_words), t2i_sim.shape[1]))
        for i, best_idx in enumerate(all_candidates):
            # keep the best index
            remain_idx = set(range(t2i_sim.shape[1])) - set(best_idx.tolist())
            best_idxs[i] = torch.tensor(list(best_idx) + list(remain_idx))
        best_idxs = best_idxs.to(torch.long)

        # calculate evaluation indices
        mod_t2i_sim = torch.zeros_like(t2i_sim)
        # Assign scores to the modified t2i_sim based on the best indices
        for i in range(best_idxs.shape[0]):
            for j in range(top_k):
                mod_t2i_sim[i, best_idxs[i, j]] = top_k - j

        eval_results = self.eval_calculator.run(
            mod_t2i_sim,
            labels_list,
            retrieval_results.image_feats,
            ext_data,
            direction,
            spice_mat_path,
        )

        # create task result
        task_result = TaskResult(
            org_indices=original_results,
            eval_indices=eval_results,
            candidates=all_candidates,
            t2i_sim=t2i_sim,
        )

        # cache the results
        if self.do_cache_sim and cached_result is None:
            self.cache_rerank(task_result, div_method, direction)

        # return the results
        return task_result
