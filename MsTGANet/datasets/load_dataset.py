import os
from typing import Optional, Callable

import pandas
from pandas import DataFrame
from torch import Tensor
from torchvision.io import read_image

from datasets.base_dataset import BaseDataset


class LoadDataset(BaseDataset):
    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self._image_names: DataFrame = pandas.read_csv(csv_file, header=0)
        self._root_dir: str = root_dir
        self._transform: Optional[Callable[[Tensor], Tensor]] = transform
    
    def __len__(self) -> int:
        return len(self._image_names)
    
    def __getitem__(self, index: int) -> (Tensor, Tensor):
        img_path = os.path.join(self._root_dir, self._image_names.iloc[index, 0])
        image = read_image(img_path)
        label = self._image_names.iloc[index, 1]
        if self._transform:
            image = self._transform(image)
        return image, label
