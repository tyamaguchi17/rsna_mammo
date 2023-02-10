import random
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from pfio.cache import MultiprocessFileCache
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

# from src.utils.directory_in_zip import DirectoryInZip
warnings.filterwarnings("ignore")

META_DATA_LIST = [
    "biopsy",
    "invasive",
    "age",
    "site_id",
    "machine_id",
    "machine_id_enc",
]
MACHINE_ID_ENCODER = {
    49: 0,
    48: 1,
    29: 2,
    21: 3,
    93: 4,
    216: 5,
    210: 6,
    170: 7,
    190: 8,
    197: 9,
    -1: 10,
}


class RSNADataset(Dataset):
    """Dataset class for rsna mammo."""

    ROOT_PATH = Path("../data")

    @classmethod
    def create_dataframe(
        cls,
        num_folds: int,
        seed: int = 2023,
        num_records: int = 0,
        fold_path: Optional[str] = "./fold/train_with_fold.csv",
        data_type: str = "train",
    ) -> pd.DataFrame:
        root = cls.ROOT_PATH

        if data_type == "train":
            if fold_path is not None:
                df = pd.read_csv(fold_path)
                if num_records:
                    df = df[:num_records]
                return df
            else:
                df = pd.read_csv(str(root / "train.csv"))
        elif data_type == "test":
            df = pd.read_csv(str(root / "sample_submission.csv"))
            return df

        n_splits = num_folds
        shuffle = True

        kfold = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        X = df["image_id"].values
        y = df["cancer"].values
        group = df["patient_id"].values
        fold = -np.ones(len(df))
        for i, (_, indices) in enumerate(kfold.split(X, y, group=group)):
            fold[indices] = i

        df["fold"] = fold

        if num_records:
            df = df[::num_records]

        return df

    def __init__(
        self,
        df: pd.DataFrame,
        phase="train",
        cfg=None,
        data_name="rsna",
    ) -> None:
        self.df = df.copy()
        self.df["original_index"] = df.index

        self.df.reset_index(inplace=True)

        lat_diff, view_diff = self.get_pair_image_ids_columns(self.df)
        self.df["lat_diff_image_ids"] = lat_diff
        self.df["view_diff_image_ids"] = view_diff
        self.df["age"] = self.df["age"].fillna(60)
        self.df = self.df.fillna(0)

        self.data_name = "rsna"
        if data_name == "rsna":
            self.df["machine_id_enc"] = self.df["machine_id"].map(MACHINE_ID_ENCODER)
            self.patient_dict = self.get_image_patient_map(self.df)
            self.idx_dict = self.get_image_idx_map(self.df)

        self.root = self.ROOT_PATH
        self.phase = phase
        self.cfg_aug = cfg.augmentation
        self.roi_th = cfg.roi_th
        self.roi_buffer = cfg.roi_buffer
        self.use_multi = cfg.use_multi

        if cfg.use_cache:
            cache_dir = "/tmp/rsna/"
            self._cache = MultiprocessFileCache(
                len(self), dir=cache_dir, do_pickle=True
            )
        else:
            cache_dir = None
            self._cache = None

    def __len__(self) -> int:
        return len(self.df)

    def get_pair_image_ids_columns(self, df):
        patient_view_dict = {}
        for idx, row in df.iterrows():
            patient_id = row["patient_id"]
            image_id = row["image_id"]
            lat = row["laterality"]
            view = row["view"]
            if patient_id not in patient_view_dict:
                patient_view_dict[patient_id] = []
            patient_view_dict[patient_id].append([view, lat, image_id])

        lat_diff = []
        view_diff = []
        for idx, row in df.iterrows():
            patient_id = row["patient_id"]
            image_id = row["image_id"]
            lat = row["laterality"]
            view = row["view"]
            _lat_diff = [
                _image_id
                for _view, _lat, _image_id in patient_view_dict[patient_id]
                if _lat != lat
            ]
            _view_diff = [
                _image_id
                for _view, _lat, _image_id in patient_view_dict[patient_id]
                if _lat == lat and _view != view
            ]
            if len(_lat_diff) == 0:
                _lat_diff = [image_id]
            if len(_view_diff) == 0:
                _view_diff = [image_id]
            lat_diff.append(_lat_diff)
            view_diff.append(_view_diff)

        return lat_diff, view_diff

    def get_image_patient_map(self, df):
        res = {}
        for idx, row in df.iterrows():
            patient_id = row["patient_id"]
            image_id = row["image_id"]
            res[image_id] = patient_id
        return res

    def get_image_idx_map(self, df):
        res = {}
        for idx, row in df.iterrows():
            image_id = row["image_id"]
            res[image_id] = idx
        return res

    def get_meta_data(self, idx):
        res = {}
        data = self.df.iloc[idx]
        for meta in META_DATA_LIST:
            res[meta] = int(data[meta])
        res["age_scaled"] = np.float16(data[meta] / 90)
        return res

    def _read_image(self, index):
        root = self.ROOT_PATH
        image_id = self.df.at[index, "image_id"]
        if self.data_name == "rsna":
            patient_id = self.patient_dict[image_id]
            path = root / f"{patient_id}/{image_id}.png"
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image, 0

    def read_image(self, index):
        if self._cache:
            image, _ = self._cache.get_and_cache(index, self._read_image)
        else:
            image, _ = self._read_image(index)
        return image

    def augmentation_2(self, image_1, image_2):
        cfg = self.cfg_aug
        if random.uniform(0, 1) < cfg.p_shuffle_view:
            image_1, image_2 = image_2, image_1
        view_1_dup_flag = random.uniform(0, 1) < cfg.p_dup_view_1
        view_2_dup_flag = random.uniform(0, 1) < cfg.p_dup_view_2
        if view_1_dup_flag ^ view_2_dup_flag:
            if view_1_dup_flag:
                image_1, image_2 = image_1, image_1
            if view_2_dup_flag:
                image_1, image_2 = image_2, image_2
        view_1_mask_flag = random.uniform(0, 1) < cfg.p_mask_view_1
        view_2_mask_flag = random.uniform(0, 1) < cfg.p_mask_view_2
        if view_1_mask_flag ^ view_2_mask_flag:
            if view_1_mask_flag:
                image_1 *= 0
            if view_2_mask_flag:
                image_2 *= 0
        return image_1, image_2

    def augmentation(self, images):
        if len(images) == 2:
            image_1, image_2 = images
            image_1, image_2 = self.augmentation_2(image_1, image_2)
            images = [image_1, image_2]
        else:
            assert 0 == 1

        return images

    def get_multi_view_ids(self, index):
        image_id_view_1 = self.df.at[index, "image_id"]
        if self.phase == "train":
            image_id_view_2 = random.choice(self.df.at[index, "view_diff_image_ids"])
        else:
            image_id_view_2 = self.df.at[index, "view_diff_image_ids"][0]
        image_id_view_2 = np.int64(image_id_view_2)
        return image_id_view_1, image_id_view_2

    def get_different_lat_ids(self, index):
        if self.phase == "train":
            image_id_view_3 = random.choice(self.df.at[index, "lat_diff_image_ids"])
        else:
            image_id_view_3 = self.df.at[index, "lat_diff_image_ids"][0]
        index_2 = self.idx_dict[image_id_view_3]
        image_id_view_3, image_id_view_4 = self.get_multi_view_ids(index_2)
        return image_id_view_3, image_id_view_4

    def get_roi_crop(self, image, threshold=0.1, buffer=30):
        y_max, x_max = image.shape
        image2 = image > image.mean()
        y_mean = image2.mean(1)
        x_mean = image2.mean(0)
        x_mean[:5] = 0
        x_mean[-5:] = 0
        y_mean[:5] = 0
        y_mean[-5:] = 0
        y_mean = (y_mean - y_mean.min() + 1e-4) / (y_mean.max() - y_mean.min() + 1e-4)
        x_mean = (x_mean - x_mean.min() + 1e-4) / (x_mean.max() - x_mean.min() + 1e-4)
        y_slice = np.where(y_mean > threshold)[0]
        x_slice = np.where(x_mean > threshold)[0]
        if len(x_slice) == 0:
            x_start, x_end = 0, x_max
        else:
            x_start, x_end = max(x_slice.min() - buffer, 0), min(
                x_slice.max() + buffer, x_max
            )
        if len(y_slice) == 0:
            y_start, y_end = 0, y_max
        else:
            y_start, y_end = max(y_slice.min() - buffer, 0), min(
                y_slice.max() + buffer, y_max
            )
        return x_start, y_start, x_end, y_end

    def get_bbox_aug(self, image):
        cfg = self.cfg_aug
        if random.uniform(0, 1) < cfg.p_th:
            th = random.uniform(cfg.roi_th_min, cfg.roi_th_max)
        else:
            th = self.roi_th
        buffer = self.roi_buffer
        x_min, y_min, x_max, y_max = self.get_roi_crop(
            image, threshold=th, buffer=buffer
        )
        if random.uniform(0, 1) < cfg.p_crop_resize:
            crop_scale = random.uniform(
                cfg.bbox_size_scale_min, cfg.bbox_size_scale_max
            )
            x_g = (x_min + x_max) / 2
            y_g = (y_min + y_max) / 2
            dx = x_max - x_min
            dy = y_max - y_min
            dx *= crop_scale
            dy *= crop_scale
            x_min = max(0, int(x_g - dx / 2))
            y_min = max(0, int(y_g - dy / 2))
            x_max = min(image.shape[1], int(x_g + dx / 2))
            y_max = min(image.shape[0], int(y_g + dy / 2))
        return x_min, y_min, x_max, y_max

    def get_bbox(self, image):
        th = self.roi_th
        buffer = self.roi_buffer
        return self.get_roi_crop(image, threshold=th, buffer=buffer)

    def __getitem__(self, index: int):

        image_id_view_1, image_id_view_2 = self.get_multi_view_ids(index)
        label = self.df.loc[index, "cancer"]
        patient_id = self.df.loc[index, "patient_id"]
        laterality = self.df.loc[index, "laterality"]

        idx_view_1 = self.idx_dict[image_id_view_1]
        idx_view_2 = self.idx_dict[image_id_view_2]
        image_1 = self.read_image(idx_view_1)
        if self.use_multi:
            image_2 = self.read_image(idx_view_2)

        if self.phase == "train":
            x_min, y_min, x_max, y_max = self.get_bbox_aug(image_1)
            image_1 = image_1[y_min:y_max, x_min:x_max]
            if self.use_multi:
                x_min, y_min, x_max, y_max = self.get_bbox_aug(image_2)
                image_2 = image_2[y_min:y_max, x_min:x_max]
                image_1, image_2 = self.augmentation([image_1, image_2])
        else:
            x_min, y_min, x_max, y_max = self.get_bbox(image_1)
            image_1 = image_1[y_min:y_max, x_min:x_max]
            if self.use_multi:
                x_min, y_min, x_max, y_max = self.get_bbox(image_2)
                image_2 = image_2[y_min:y_max, x_min:x_max]

        meta_data = self.get_meta_data(index)

        res = {
            "original_index": self.df.at[index, "original_index"],
            "image_id": image_id_view_1,
            "image_id_2": image_id_view_2,
            "patient_id": patient_id,
            "laterality": {"L": 0, "R": 1}[laterality],
            "label": label,
            "image_1": image_1,
        }
        if self.use_multi:
            res.update(
                {
                    "image_2": image_2,
                }
            )
        res.update(meta_data)
        # res["label_2"] = res["label"]
        # res["biopsy_2"] = res["biopsy"]
        # res["invasive_2"] = res["invasive"]

        return res
