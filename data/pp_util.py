import contextlib
import json
import time
import warnings
from ast import literal_eval
from pathlib import Path
from ssl import SSLError

import datasets
import numpy as np
import requests
import torch
from datasets import concatenate_datasets
from dateutil import parser
from PIL import Image
from PIL.Image import UnidentifiedImageError

from msdpp.schema import RetrievalDataset

warnings.filterwarnings("ignore")


class PixelProsePreprocess:
    def __init__(self, img_save_root: Path) -> None:
        self.img_save_root = img_save_root

    def filter_datetime(self, data: dict) -> bool:
        exif = data["exif"]
        if exif is None:
            return False
        if "EXIF DateTimeOriginal" not in exif:
            return False

        datetime_str = json.loads(exif)["EXIF DateTimeOriginal"]
        if datetime_str == "":
            return False

        x = " ".join(
            [
                datetime_str.split(" ")[0].replace(":", "-"),
                *datetime_str.split(" ")[1:],
            ]
        ).split(".")[0]
        # check the format of the date
        try:
            y = parser.parse(x)
        except:  # noqa: E722
            return False

        return not (y.year < 100 or y.year > 2024)  # noqa: PLR2004

    def filter_gps(self, data: dict) -> bool:
        exif = data["exif"]
        if exif is None:
            return False
        exif = json.loads(exif)

        if not ("GPS GPSLatitude" in exif and "GPS GPSLongitude" in exif):
            return False

        if not ("GPS GPSLatitudeRef" in exif and "GPS GPSLongitudeRef" in exif):
            return False

        if (
            exif["GPS GPSLatitude"] == "[0, 0, 0]"
            or exif["GPS GPSLongitude"] == "[0, 0, 0]"
        ):
            return False

        try:
            lat = literal_eval(exif["GPS GPSLatitude"])
            long = literal_eval(exif["GPS GPSLongitude"])
            if isinstance(lat, list) and len(lat) == 0:
                return False
            if isinstance(long, list) and len(long) == 0:
                return False

            if isinstance(lat, tuple) and (len(lat) == 0 or lat[0] == 0):
                return False
            if isinstance(long, tuple) and (len(long) == 0 or long[0] == 0):
                return False
        except ZeroDivisionError:
            return False
        except ValueError:
            return False

        return True

    # convert the above latitude and longitude to decimal
    def dms_to_dd(self, dms: str) -> float:
        dms_float_list = literal_eval(dms)

        if isinstance(dms_float_list, float):
            d = dms_float_list
            m = 0
            s = 0
        elif len(dms_float_list) == 3:  # noqa: PLR2004
            d, m, s = dms_float_list
        elif len(dms_float_list) == 2:  # noqa: PLR2004
            d, m = dms_float_list
            s = 0
        else:
            msg = f"Invalid dms format: {dms_float_list}, dms: {dms}"

            raise ValueError(msg)
        return d + m / 60 + s / 3600

    def cvt_gps(self, data: dict) -> dict | None:
        exif = json.loads(data["exif"])
        lat_str, lon_str = exif["GPS GPSLatitude"], exif["GPS GPSLongitude"]
        try:
            lat, lon = (
                self.dms_to_dd(lat_str),
                self.dms_to_dd(lon_str),
            )  # (34.038, 118.0575)
        except TypeError:
            return None

        if exif["GPS GPSLatitudeRef"] == "S":
            lat = -lat
        if exif["GPS GPSLongitudeRef"] == "W":
            lon = -lon

        data["gps"] = [lat, lon]

        # convert latitude and longitude to x, y
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # convert latitude and longitude to x, y
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        data["gps_xyz"] = [x, y, z]
        return data

    def cvt_datetime(self, data: dict) -> dict | bool:
        exif = json.loads(data["exif"])
        datetime_str = exif["EXIF DateTimeOriginal"]
        if datetime_str == "":
            return False

        x = " ".join(
            [
                datetime_str.split(" ")[0].replace(":", "-"),
                *datetime_str.split(" ")[1:],
            ]
        ).split(".")[0]
        # check the format of the date
        y = parser.parse(x)
        data["year"] = y.year
        data["month"] = y.month
        data["hour"] = y.hour
        data["minute"] = y.minute

        return data

    def remove_duplicates(self, all_dataset: datasets.Dataset) -> datasets.Dataset:
        uid = all_dataset["uid"]
        url = all_dataset["url"]
        img_names = [url[i].split("/")[-1] for i in range(len(url))]
        duplicates = []
        dup_url = []
        # print urls with same uid
        for i in range(len(uid) - 1):
            idx = []
            # get the index of the same uid
            for j in range(i + 1, len(uid)):
                if i == j:
                    continue
                if (uid[i] == uid[j] and img_names[i] == img_names[j]) or url[
                    i
                ] == url[j]:
                    idx.append(j)

            if len(idx) > 0:
                dup_url.append(url[i])
                duplicates.extend(idx)

        delete_idx = set(duplicates)
        all_idx = set(range(len(all_dataset)))

        return all_dataset.select(list(all_idx - delete_idx))

    def fix_orientation(self, img: Image.Image) -> Image.Image:
        f = {
            0: lambda img: img,
            1: lambda img: img,
            2: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda img: img.transpose(Image.Transpose.ROTATE_180),
            4: lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_90
            ),
            6: lambda img: img.transpose(Image.Transpose.ROTATE_270),
            7: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_270
            ),
            8: lambda img: img.transpose(Image.Transpose.ROTATE_90),
        }

        if not hasattr(img, "_getexif"):
            return img

        exif = img._getexif()  # pyright: ignore[reportAttributeAccessIssue]

        if exif is None:
            return img

        orientation = exif.get(0x112, 1)

        fixed_img: Image.Image = f[orientation](img)
        return fixed_img

    def dl_img(self, url: str) -> Image.Image | None:
        max_size = 1024
        try:
            org_img = Image.open(requests.get(url, stream=True, verify=False).raw)
            rotated_img: Image.Image = org_img
            with contextlib.suppress(KeyError):
                rotated_img = self.fix_orientation(org_img)

            img = rotated_img
            w, h = rotated_img.size
            if w > max_size or h > max_size:
                if w > h:
                    img = rotated_img.resize((max_size, int(h / w * max_size)))
                else:
                    img = rotated_img.resize((int(w / h * max_size), max_size))

            img = img.convert("RGB")
        except UnidentifiedImageError:
            img = None
        except SSLError:
            img = None
        except requests.exceptions.SSLError:
            img = None
        except requests.exceptions.ProxyError:
            img = None
        except requests.exceptions.ConnectionError:
            img = None
        except requests.exceptions.TooManyRedirects:
            img = None
        except:
            img = None

        return img

    def dl_and_save_img(self, data: dict) -> dict:
        img_root = self.img_save_root
        url = data["url"]
        key = data["uid"] + "_" + url.split("/")[-1].split(".")[0][:]
        img_path = img_root / f"{key}.jpg"
        if img_path.exists():
            data["image"] = f"{key}.jpg"
            return data
        img = None
        for _ in range(3):
            img = self.dl_img(url)
            if img is not None:
                break
            time.sleep(3)

        if img is not None:
            try:
                img.save(img_path, "JPEG")
                data["image"] = f"{key}.jpg"
            except OSError:  # Filename too long
                img_path = img_root / f"{key[:100]}.jpg"
                img.save(img_path, "JPEG")
                data["image"] = f"{key[:100]}.jpg"
        else:
            data["image"] = "-1"
        return data

    def add_image(self, data: dict) -> dict:
        img_path = self.img_save_root / f"{data['image']}"
        img = Image.open(img_path)
        data["image"] = img
        return data

    def filter_img(self, data: dict) -> bool:
        return data["image"] != "-1"

    def filter_gps_outlier(self, data: dict) -> bool:
        if data["gps"][0] < -90 or data["gps"][0] > 90:
            return False
        if data["gps"][1] < -180 or data["gps"][1] > 180:
            return False

        return True

    def get_caption_label(
        self, dataset: datasets.Dataset | datasets.DatasetDict, n_caption: int
    ) -> tuple[list[str], dict]:
        # select 1000 captions and generate label
        captions = dataset["vlm_caption"]
        torch.random.set_rng_state(torch.Generator().manual_seed(42).get_state())
        target_ids = torch.randperm(len(captions))[:n_caption].tolist()

        target_captions = [captions[i].replace("\n", "") for i in target_ids]

        label = {}
        for i, caption in enumerate(target_captions):
            label[caption] = torch.zeros(len(captions), dtype=torch.int)
            label[caption][target_ids[i]] = 1

        return target_captions, label

    def val_test_ext_split(
        self,
        val_dataset: datasets.Dataset,
        val_captions: list[str],
        test_dataset: datasets.Dataset,
        test_captions: list[str],
        val_label: dict,
        test_label: dict,
        ext: torch.Tensor | list,
        name: str,
        target_ids: list[int],
        n_val: int,
    ) -> tuple[RetrievalDataset, RetrievalDataset]:
        train_idx = target_ids[:n_val]
        test_idx = target_ids[n_val:]

        if isinstance(ext, torch.Tensor):
            train_ext_data = ext[train_idx]
            test_ext_data = ext[test_idx]
        else:
            train_ext_data = [e[train_idx] for e in ext]
            test_ext_data = [e[test_idx] for e in ext]

        val_dataset_ret = RetrievalDataset(
            name + "_val",
            val_dataset,
            val_captions,
            val_label,
            train_ext_data,
        )

        test_dataset_ret = RetrievalDataset(
            name + "_test",
            test_dataset,
            test_captions,
            test_label,
            test_ext_data,
        )

        return val_dataset_ret, test_dataset_ret

    def filter_and_save(
        self, dataset: datasets.DatasetDict, save_path: Path
    ) -> datasets.Dataset:
        filtered = dataset.filter(self.filter_datetime).filter(self.filter_gps)

        all_dataset: datasets.Dataset = (
            concatenate_datasets(
                [filtered["cc12m"], filtered["commonpool"], filtered["redcaps"]]
            )
            .map(self.cvt_gps)
            .map(self.cvt_datetime)
        )
        dataset_wo_dup = self.remove_duplicates(all_dataset)

        dataset_wo_dup.save_to_disk(save_path)

        return dataset_wo_dup

    def dl_and_add_img(
        self, dataset: datasets.Dataset, num_proc: int = 80
    ) -> datasets.Dataset:
        """Download images and add them to the dataset."""
        return (
            dataset.map(self.dl_and_save_img, num_proc=num_proc)
            .filter(
                self.filter_img, num_proc=num_proc
            )  # Filter out images that failed to download
            .map(self.add_image, num_proc=num_proc)
            .cast_column("image", datasets.Image())
            .filter(self.filter_gps_outlier, num_proc=num_proc)
        )

    def split_and_save_val_test(
        self,
        val_dataset: datasets.Dataset,
        val_captions: list[str],
        test_dataset: datasets.Dataset,
        test_captions: list[str],
        val_labels: dict,
        test_labels: dict,
        ext_data: torch.Tensor | list,
        name: str,
        target_ids: list[int],
        n_val: int,
        val_path: Path,
        test_path: Path,
    ) -> None:
        val_hour, test_hour = self.val_test_ext_split(
            val_dataset,
            val_captions,
            test_dataset,
            test_captions,
            val_labels,
            test_labels,
            ext_data,
            name,
            target_ids,
            n_val,
        )

        with val_path.open("wb") as f:
            torch.save(val_hour, f)

        with test_path.open("wb") as f:
            torch.save(test_hour, f)

    def split_and_save(
        self,
        dataset: datasets.Dataset,
        ext_data_dict: dict[str, torch.Tensor | list],
        save_root: Path,
        n_val_query: int = 50,
        n_test_query: int = 1000,
    ) -> None:
        """Split the dataset into validation and test sets."""
        torch.random.set_rng_state(torch.Generator().manual_seed(42).get_state())
        target_ids = torch.randperm(len(dataset)).tolist()
        n_val = int(len(dataset) * 0.2)

        val_dataset = dataset.select(target_ids[:n_val])
        test_dataset = dataset.select(target_ids[n_val:])

        val_captions, val_labels = self.get_caption_label(val_dataset, n_val_query)
        test_captions, test_labels = self.get_caption_label(
            test_dataset, n_test_query
        )

        for k, ext_data in ext_data_dict.items():
            self.split_and_save_val_test(
                val_dataset,
                val_captions,
                test_dataset,
                test_captions,
                val_labels,
                test_labels,
                ext_data,
                k,
                target_ids,
                n_val,
                val_path=save_root / f"{k}_val.pkl",
                test_path=save_root / f"{k}_test.pkl",
            )
