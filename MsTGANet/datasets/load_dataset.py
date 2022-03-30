import os
from typing import Optional, Callable

from torch import Tensor
from torchvision.io import read_image

import imageio as iio

from datasets.base_dataset import BaseDataset


class LoadDataset(BaseDataset):
    def __init__(self, images: str, labels: str, transform: Optional[Callable[[Tensor], Tensor]] = None,
                 label_transform: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self._images = images
        self._labels = labels
        self._image_names = os.listdir(images)
        self._label_names = os.listdir(labels)
        self._transform: Optional[Callable[[Tensor], Tensor]] = transform
        self._label_transform: Optional[Callable[[Tensor], Tensor]] = label_transform

    def __len__(self) -> int:
        return len(self._image_names)

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        image_path = os.path.join(self._images, self._image_names[index])
        image = iio.imread(image_path)
        label_path = os.path.join(self._images, self._label_names[index])
        label = iio.imread(label_path)
        if self._transform:
            image = self._transform(image)
        if self._label_transform:
            label = self._transform(label)
        return image, label
