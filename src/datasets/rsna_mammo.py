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
    "BIRADS",
    "difficult_negative_case",
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
        num_folds: int = 4,
        seed: int = 2023,
        num_records: int = 0,
        fold_path: Optional[str] = "./fold/train_with_fold.csv",
        data_type: str = "train",
        pl_path: Optional[str] = None,
        rsna_bbox_path: Optional[str] = "./bboxes/rsna/det_result_001_baseline.csv",
        vindr_bbox_path: Optional[
            str
        ] = "./bboxes/vindr/det_result_vindr_001_baseline.csv",
    ) -> pd.DataFrame:
        root = cls.ROOT_PATH

        if data_type == "train":
            if fold_path is not None:
                df = pd.read_csv(fold_path)
                df_bbox = pd.read_csv(rsna_bbox_path)
                df_bbox["patient_id"] = (
                    df_bbox["name"].map(lambda x: x.split("/")[0]).astype(int)
                )
                df_bbox["image_id"] = (
                    df_bbox["name"]
                    .map(lambda x: x.split("/")[1].replace(".png", ""))
                    .astype(int)
                )
                df = df.merge(df_bbox, on=["patient_id", "image_id"])
                df.loc[df["cancer"] == 1, "BIRADS"] = 3
                df.loc[np.isnan(df["BIRADS"]), "BIRADS"] = 1
                if num_records:
                    df = df[:num_records]
                df["is_rsna"] = 1
                return df
            else:
                # not supported
                df = pd.read_csv(str(root / "train.csv"))
                assert 0 == 1
        elif data_type == "vindr":
            if pl_path is None:
                df = pd.read_csv("./vindr/vindr_train.csv")
                df["cancer"] = 0
            else:
                df = pd.read_csv(pl_path)
            df_bbox = pd.read_csv(vindr_bbox_path)
            df_bbox["patient_id"] = (
                df_bbox["name"].map(lambda x: x.split("_")[0]).astype(str)
            )
            df_bbox["image_id"] = (
                df_bbox["name"]
                .map(lambda x: x.split("_")[1].replace(".png", ""))
                .astype(str)
            )
            df = df.merge(df_bbox, on=["patient_id", "image_id"])
            df["biopsy"] = 0
            df["invasive"] = 0
            df["site_id"] = 0
            df["machine_id"] = -1
            df["is_rsna"] = 0
            return df

        elif data_type == "test":
            # not supported
            df = pd.read_csv(str(root / "sample_submission.csv"))
            assert 0 == 1
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

        self.data_name = data_name
        if data_name == "rsna" or data_name == "vindr":
            self.df["machine_id_enc"] = self.df["machine_id"].map(MACHINE_ID_ENCODER)
            self.patient_dict = self.get_image_patient_map(self.df)
            self.idx_dict = self.get_image_idx_map(self.df)

        self.root = self.ROOT_PATH
        self.phase = phase
        self.cfg_aug = cfg.augmentation
        self.roi_th = cfg.roi_th
        self.roi_buffer = cfg.roi_buffer
        self.use_multi_view = cfg.use_multi_view
        self.use_multi_lat = cfg.use_multi_lat
        self.use_yolo = cfg.use_yolo

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
        res["age_scaled"] = data["age"] / 90
        res["BIRADS_scaled"] = data["BIRADS"] / 5
        return res

    def _read_image(self, index):
        root = self.ROOT_PATH
        image_id = self.df.at[index, "image_id"]
        if self.data_name == "rsna" or self.data_name == "vindr":
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
        image_id_view_2 = image_id_view_2
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

    def get_bbox_aug(self, image, index):
        cfg = self.cfg_aug
        if random.uniform(0, 1) < cfg.p_th:
            th = random.uniform(cfg.roi_th_min, cfg.roi_th_max)
        else:
            th = self.roi_th
        buffer = self.roi_buffer
        if random.uniform(0, 1) < cfg.p_roi_crop:
            x_min, y_min, x_max, y_max = self.get_roi_crop(
                image, threshold=th, buffer=buffer
            )
        else:
            x_min, y_min, x_max, y_max = self.df[
                ["ymin", "xmin", "ymax", "xmax"]
            ].values[index]
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

    def get_bbox(self, image, index):
        th = self.roi_th
        buffer = self.roi_buffer
        if self.use_yolo:
            x_min, y_min, x_max, y_max = self.df[
                ["ymin", "xmin", "ymax", "xmax"]
            ].values[index]
        else:
            x_min, y_min, x_max, y_max = self.get_roi_crop(
                image, threshold=th, buffer=buffer
            )
        return x_min, y_min, x_max, y_max

    def __getitem__(self, index: int):

        image_id_view_1, image_id_view_2 = self.get_multi_view_ids(index)
        label = self.df.loc[index, "cancer"]
        patient_id = self.df.loc[index, "patient_id"]
        laterality = self.df.loc[index, "laterality"]

        idx_view_1 = self.idx_dict[image_id_view_1]
        idx_view_2 = self.idx_dict[image_id_view_2]

        image_id_view_3, image_id_view_4 = self.get_different_lat_ids(index)
        idx_view_3 = self.idx_dict[image_id_view_3]
        idx_view_4 = self.idx_dict[image_id_view_4]

        meta_data = self.get_meta_data(idx_view_1)
        label_2 = self.df.loc[idx_view_3, "cancer"]
        laterality_2 = self.df.loc[idx_view_3, "laterality"]
        meta_data_2 = self.get_meta_data(idx_view_3)

        image_1 = self.read_image(idx_view_1)
        if self.use_multi_view:
            image_2 = self.read_image(idx_view_2)

        if self.phase == "train":
            x_min, y_min, x_max, y_max = self.get_bbox_aug(image_1, idx_view_1)
            image_1 = image_1[y_min:y_max, x_min:x_max]
            if self.use_multi_view:
                x_min, y_min, x_max, y_max = self.get_bbox_aug(image_2, idx_view_2)
                image_2 = image_2[y_min:y_max, x_min:x_max]
                image_1, image_2 = self.augmentation([image_1, image_2])
            if self.use_multi_lat:
                image_id_view_3, image_id_view_4 = self.get_different_lat_ids(index)
                idx_view_3 = self.idx_dict[image_id_view_3]
                idx_view_4 = self.idx_dict[image_id_view_4]
                image_3 = self.read_image(idx_view_3)
                image_4 = self.read_image(idx_view_4)
                x_min, y_min, x_max, y_max = self.get_bbox_aug(image_3, idx_view_3)
                image_3 = image_3[y_min:y_max, x_min:x_max]
                x_min, y_min, x_max, y_max = self.get_bbox_aug(image_4, idx_view_4)
                image_4 = image_4[y_min:y_max, x_min:x_max]
        else:
            x_min, y_min, x_max, y_max = self.get_bbox(image_1, idx_view_1)
            image_1 = image_1[y_min:y_max, x_min:x_max]
            if self.use_multi_view:
                x_min, y_min, x_max, y_max = self.get_bbox(image_2, idx_view_2)
                image_2 = image_2[y_min:y_max, x_min:x_max]
            if self.use_multi_lat:
                image_3 = self.read_image(idx_view_3)
                image_4 = self.read_image(idx_view_4)
                x_min, y_min, x_max, y_max = self.get_bbox(image_3, idx_view_3)
                image_3 = image_3[y_min:y_max, x_min:x_max]
                x_min, y_min, x_max, y_max = self.get_bbox(image_4, idx_view_4)
                image_4 = image_4[y_min:y_max, x_min:x_max]

        if self.use_multi_lat:
            if self.phase == "train":
                if random.uniform(0, 1) < self.cfg_aug.p_shuffle_lat:
                    image_1, image_2, image_3, image_4 = (
                        image_3,
                        image_4,
                        image_1,
                        image_2,
                    )
                    label, label_2 = label_2, label
                    laterality, laterality_2 = laterality_2, laterality
                    meta_data, meta_data_2 = meta_data_2, meta_data
        laterality = {"L": 0, "R": 1}[laterality]
        laterality_2 = {"L": 0, "R": 1}[laterality_2]
        res = {
            "original_index": self.df.at[index, "original_index"],
            "image_id": image_id_view_1,
            "image_id_2": image_id_view_2,
            "image_id_3": image_id_view_3,
            "image_id_4": image_id_view_4,
            "patient_id": patient_id,
            "laterality": laterality,
            "laterality_2": laterality_2,
            "label": label,
            "label_2": label_2,
            "image_1": image_1,
        }
        if self.use_multi_view:
            res.update(
                {
                    "image_2": image_2,
                }
            )
        res.update(meta_data)
        if self.use_multi_lat:
            res.update({key + "_" + "2": meta_data_2[key] for key in meta_data_2})
            res.update(
                {
                    "image_3": image_3,
                    "image_4": image_4,
                }
            )

        return res
