from abc import ABC, abstractmethod

from typing import Tuple, Optional

from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    # The first output is the image. The second output is the label. If there is no label, return None.
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]: ...
    
    @abstractmethod
    def __len__(self) -> int: ...
