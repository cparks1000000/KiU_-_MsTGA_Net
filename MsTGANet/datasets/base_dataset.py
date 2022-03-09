import \
	os
from abc import ABC, abstractmethod

from typing import Tuple, Optional

from torch import Tensor
from torch.utils.data import Dataset

from torchvision.io import read_image
import pandas as pd


class BaseDataset(Dataset, ABC):
	# The first output is the image. The second output is the label. If there is no label, return None.
	@abstractmethod
	def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]: ...

	@abstractmethod
	def __len__(self) -> int: ...


class LoadDataset(BaseDataset):

	def __init__(self, csv_file, root_dir, transform=None):
		self.data_labels = pd.read_csv(csv_file, header=0)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.data_labels)

	def __getitem__(self, index):
		img_path = os.path.join(self.root_dir, self.data_labels.iloc[index, 0])
		image = read_image(img_path)
		label = self.data_labels.iloc[index, 1]
		if self.transform:
			image = self.transform(image)
		return [image, label]
