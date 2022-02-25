from typing import TypeVar, NewType, Union, Tuple, List, Optional

from torch import nn as nn, Tensor
from torch.nn.functional import interpolate

from MsTGANet.modules.convolutions import ConvolutionBlock


T = TypeVar('T')
Flex = NewType('Multiple', Union[T, Tuple[T, ...]])


class UpConvolution(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super(UpConvolution, self).__init__()
        # noinspection PyArgumentEqualDefault
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvolutionBlock(channels_in, channels_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest', align_corners: bool = None) -> None:
        super().__init__()
        self._size = size
        self._scale_factor = scale_factor
        self._mode = mode
        self._align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        return interpolate( x, size=self._size, scale_factor=self._scale_factor, mode=self._mode, align_corners=self._align_corners )