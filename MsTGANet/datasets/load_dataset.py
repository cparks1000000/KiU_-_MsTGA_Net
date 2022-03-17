import \
    os
from typing import Optional, Callable

import \
    pandas
from torch import Tensor
from torchvision.io import read_image

from datasets.base_dataset import BaseDataset


class LoadDataset(BaseDataset):
	def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable[[Tensor], Tensor]] = None) -> None:
		self.data_labels = pandas.read_csv(csv_file, header=0)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self) -> int:
		return len(self.data_labels)

	def __getitem__(self, index: int) -> (Tensor, Tensor):
		img_path = os.path.join(self.root_dir, self.data_labels.iloc[index, 0])
		image = read_image(img_path)
		label = self.data_labels.iloc[index, 1]
		if self.transform:
			image = self.transform(image)
		return image, label
