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
    ):
        self.base = base
        self.transform = transform
        self.view = view
        self.idx = base.df[base.df["view"] == view].index

    def __len__(self) -> int:
        if self.view is None:
            return len(self.base)
        else:
            return len(self.idx)

    def apply_transform(self, data):

        image_1 = data.pop("image_1")
        if self.base.use_multi_view:
            image_2 = data.pop("image_2")

        transformed = self.transform(image=image_1)
        image_1 = transformed["image"]  # (1, H, W)
        if self.base.use_multi_view:
            transformed = self.transform(image=image_2)
            image_2 = transformed["image"]  # (1, H, W)
            data["image"] = np.concatenate([image_1, image_2])  # (2, H, W)
        else:
            data["image"] = image_1

        return data

    def __getitem__(self, index: int):
        if self.view is not None:
            index = self.idx[index]
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data
