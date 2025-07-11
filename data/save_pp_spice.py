# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from pycocoevalcap.spice.spice import Spice as _Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm import tqdm

from msdpp.schema import RetrievalDataset

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = "/msdpp/share_datasets/SPICE/target/spice-1.0.jar"
TEMP_DIR = "tmp"
CACHE_DIR = "cache"

HF_HOME = Path(os.environ.get("HF_HOME", "./"))

save_root = Path("/msdpp/share_datasets/SPICE_ver2/")

save_task_data_root = HF_HOME / "tasks"

val_dataset_path = save_task_data_root / "PP_geo_val.pkl"
test_dataset_path = save_task_data_root / "PP_geo_test.pkl"

batch_size = 1000

save_root.mkdir(parents=True, exist_ok=True)


# %%


class Spice(_Spice):  # type: ignore[misc]
    def compute_score(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, gts: list[str], res: list[str]
    ) -> tuple[float, np.ndarray, list[dict]]:
        gt_data = [{"caption": gt} for gt in gts]

        ref_data = [{"caption": ref} for ref in res]

        cwd = Path(__file__).resolve().parent
        temp_dir = cwd / TEMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)

        in_file_name = ""
        in_file_ref_name = ""
        with (
            tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir, mode="w+"
            ) as in_file,
            tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir, mode="w+"
            ) as in_file_ref,
        ):
            json.dump(gt_data, in_file, indent=2)
            json.dump(ref_data, in_file_ref, indent=2)

            in_file_name = in_file.name
            in_file_ref_name = in_file_ref.name

        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)  # noqa: SIM115
        out_file.close()

        # Start job
        cache_dir = cwd / CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        spice_cmd = [
            "java",
            "-jar",
            "-Xmx64G",
            SPICE_JAR,
            in_file_name,
            in_file_ref_name,
            "-out",
            out_file.name,
            "-subset",
            "-silent",
        ]
        subprocess.check_call(spice_cmd, cwd=cwd)  # noqa: S603

        # Read and process results
        with Path(out_file.name).open() as data_file:
            results = json.load(data_file)

        Path(in_file_name).unlink(missing_ok=True)
        Path(in_file_ref_name).unlink(missing_ok=True)
        Path(out_file.name).unlink(missing_ok=True)

        imgid_to_scores = {}
        spice_scores = []
        for item in results:
            imgid_to_scores[item["image_id"]] = item["scores"]
            spice_scores.append(self.float_convert(item["scores"]["All"]["f"]))
        average_score = np.mean(np.array(spice_scores))
        scores = []

        for result in results:
            score = self.float_convert(result["scores"]["All"]["f"])
            scores.append(score)

        scores_np = np.asarray(scores).reshape((len(gts), len(res)))
        return float(average_score), scores_np, results


spice_scorer = Spice()
tokenizer = PTBTokenizer()


for phase in ["val", "test"]:
    cache_dir = Path(__file__).resolve().parent / CACHE_DIR

    if phase == "val":
        dataset: RetrievalDataset = torch.load(val_dataset_path, weights_only=False)
    else:
        dataset = torch.load(test_dataset_path, weights_only=False)

    target_caps = dataset.dataset["vlm_caption"]
    query_caps = dataset.retrieval_words
    spice_matrix = np.zeros((len(target_caps), len(query_caps)))

    query_for_tokenize = {
        i: [{"caption": query}] for i, query in enumerate(query_caps)
    }
    res_tokenized = tokenizer.tokenize(query_for_tokenize)
    res_tokenized = [i[0] for i in list(res_tokenized.values())]

    target_for_tokenize = {
        i: [{"caption": cap}] for i, cap in enumerate(target_caps)
    }
    gtr_tokenized = tokenizer.tokenize(target_for_tokenize)
    gtr_tokenized = [i[0][:2800] for i in list(gtr_tokenized.values())]

    for i in tqdm(range(1, len(gtr_tokenized), batch_size)):
        batch_gt = gtr_tokenized[i : min(i + batch_size, len(gtr_tokenized))]

        spice, x, results = spice_scorer.compute_score(
            gts=batch_gt, res=res_tokenized
        )
        torch.save(x, cache_dir / f"pp_spice_{phase}_{i}.pt")

        gc.collect()

    spices = natsorted(cache_dir.glob(f"pp_spice_{phase}_*.pt"))

    spices_np = [torch.load(spice) for spice in spices]
    spice_pt = torch.from_numpy(spices_np[0][0, :]).unsqueeze(0)

    for spice_np in spices[1:]:
        spice_pt = torch.cat((spice_pt, torch.from_numpy(spice_np)), 0)

    torch.save(spice_pt, save_root / f"pp_spice_{phase}.pt")
