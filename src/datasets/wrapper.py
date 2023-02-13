from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
        phase: str,
        view: Optional[str] = None,
        lat: Optional[str] = None,
    ):
        self.base = base
        self.transform = transform
        self.view = view
        self.lat = lat
        self.idx = base.df[base.df["view"] == view].index
        if lat is not None:
            self.idx = base.df[base.df["laterality"] == lat].index

    def __len__(self) -> int:
        if self.view is None and self.lat is None:
            return len(self.base)
        else:
            return len(self.idx)

    def apply_transform(self, data):

        image_1 = data.pop("image_1")
        if self.base.use_multi_view:
            image_2 = data.pop("image_2")
        if self.base.use_multi_lat:
            image_3 = data.pop("image_3")
            image_4 = data.pop("image_4")

        transformed = self.transform(image=image_1)
        image_1 = transformed["image"]  # (1, H, W)
        if self.base.use_multi_view:
            transformed = self.transform(image=image_2)
            image_2 = transformed["image"]  # (1, H, W)
            if self.base.use_multi_lat:
                transformed = self.transform(image=image_3)
                image_3 = transformed["image"]  # (1, H, W)
                transformed = self.transform(image=image_4)
                image_4 = transformed["image"]  # (1, H, W)
                data["image"] = np.concatenate(
                    [image_1, image_2, image_3, image_4]
                )  # (4, H, W)
            else:
                data["image"] = np.concatenate([image_1, image_2])  # (2, H, W)
        else:
            data["image"] = image_1

        return data

    def __getitem__(self, index: int):
        if self.view is not None or self.lat is not None:
            index = self.idx[index]
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data
