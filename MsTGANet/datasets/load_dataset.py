import os
from typing import Optional, Callable

from torch import Tensor

import imageio as iio

from datasets.base_dataset import BaseDataset


class LoadDataset(BaseDataset):
    def __init__(self, images_path: str, labels_path: str, transform: Optional[Callable[[Tensor], Tensor]] = None,
                 label_transform: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self._images_path: str = images_path
        self._labels_path: str = labels_path
        self._image_names: list[str] = os.listdir(images_path)
        self._label_names: list[str] = os.listdir(labels_path)
        self._transform: Optional[Callable[[Tensor], Tensor]] = transform
        self._label_transform: Optional[Callable[[Tensor], Tensor]] = label_transform

    def __len__(self) -> int:
        return len(self._image_names)

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        image_path: str = os.path.join(self._images_path, self._image_names[index])
        label_path: str = os.path.join(self._images_path, self._label_names[index])
        image: Tensor = iio.imread(image_path)
        label: Tensor = iio.imread(label_path)
        if self._transform:
            image: Tensor = self._transform(image)
        if self._label_transform:
            label: Tensor = self._transform(label)
        return image, label