# %%

import os
import warnings
from pathlib import Path

import datasets
import torch
from pp_util import (
    PixelProsePreprocess,
)

from msdpp.data import datetime_embeds

warnings.filterwarnings("ignore")

HF_HOME = Path(os.environ.get("HF_HOME", "./"))

save_path = HF_HOME / "self_datasets" / "pixelprose" / "val"
save_filtered_path = HF_HOME / "self_datasets" / "pixelprose_filtered" / "val"

save_task_data_root = HF_HOME / "tasks"

img_save_root = Path("/msdpp/share_datasets3/pixelprose/images")

preprocessor = PixelProsePreprocess(img_save_root=img_save_root)

save_path.mkdir(parents=True, exist_ok=True)
save_filtered_path.mkdir(parents=True, exist_ok=True)
save_task_data_root.mkdir(parents=True, exist_ok=True)
img_save_root.mkdir(parents=True, exist_ok=True)

# %%

# DL and filter the dataset
# and save it to save_path
org_dataset: datasets.DatasetDict = datasets.load_dataset(
    "tomg-group-umd/pixelprose"
)
del org_dataset["train"]  # pyright: ignore[reportIndexIssue]

all_dataset: datasets.Dataset = preprocessor.filter_and_save(org_dataset, save_path)

# %%
# dl and add image data to the dataset
dataset = preprocessor.dl_and_add_img(all_dataset, num_proc=80)

# save the dataset with images
dataset.save_to_disk(save_filtered_path)

# %%


xyz = torch.tensor(dataset["gps_xyz"])
ext = torch.nn.functional.normalize(xyz, 2, -1)

hours = torch.tensor(dataset["hour"]).to(torch.float32)
minutes = torch.tensor(dataset["minute"]).to(torch.float32)
time_data = datetime_embeds(hours, minutes)

ext_data_dict = {
    "PP_geo": ext,
    "PP_hour": time_data,
    "PP_geo_hour": [ext, time_data],
}

preprocessor.split_and_save(
    dataset,
    ext_data_dict,
    save_root=save_task_data_root,
    n_val_query=50,
    n_test_query=1000,
)
