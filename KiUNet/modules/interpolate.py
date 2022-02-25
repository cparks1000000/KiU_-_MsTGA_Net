from __future__ import annotations

from torch import nn, Tensor
from torch.nn.functional import interpolate
from typing import Union, Tuple, TypeVar, NewType

T = TypeVar('T')

Flex = NewType('Multiple', Union[T, Tuple[T, ...]])


class Interpolate(nn.Module):
	def __init__(self, size: Flex[int] = None, scale_factor: Flex[float] = None, mode: str = 'nearest', align_corners: bool = None) -> None:
		super().__init__()
		self._size = size
		self._scale_factor = scale_factor
		self._mode = mode
		self._align_corners = align_corners
		
	def forward(self, x: Tensor) -> Tensor:
		return interpolate( x, size=self._size, scale_factor=self._scale_factor, mode=self._mode, align_corners=self._align_corners )