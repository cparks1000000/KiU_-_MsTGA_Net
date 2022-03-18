from torch import long
from torch import Tensor
from torch.utils.data import Dataset

from datasets.base_dataset import BaseDataset


class DatasetWrapper(BaseDataset):
    """ This is a naive dataset whose "segmentation" is just all zeros"""
    
    def __init__(self, wrapped_dataset: Dataset) -> None:
        self._wrapped_dataset = wrapped_dataset
    
    def __len__(self) -> int:
        # noinspection PyTypeChecker
        return len(self._wrapped_dataset)
    
    def __getitem__(self, index: int) -> (Tensor, Tensor):
        return self._wrapped_dataset[index][0], (0*self._wrapped_dataset[index][0]).to(long)
