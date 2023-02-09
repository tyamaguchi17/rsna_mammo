from typing import Callable

import numpy as np
from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
        phase: str,
        tta: bool = False,
    ):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def apply_transform(self, data):

        image_1 = data.pop("image_1")
        if self.base.use_multi:
            image_2 = data.pop("image_2")

        transformed = self.transform(image=image_1)
        image_1 = transformed["image"]  # (1, H, W)
        if self.base.use_multi:
            transformed = self.transform(image=image_2)
            image_2 = transformed["image"]  # (1, H, W)
            data["image"] = np.concatenate([image_1, image_2])  # (2, H, W)
        else:
            data["image"] = image_1

        return data

    def __getitem__(self, index: int):
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data
