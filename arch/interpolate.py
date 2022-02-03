from __future__ import annotations

from torch import nn, Tensor
from torch.nn.functional import interpolate
from typing import Union, Tuple


class Interpolate(nn.Module):
	def __init__( self,
		size: Union[int, Tuple[int]] = None,
		scale_factor: Union[float, Tuple[float, ...]] = None,
		mode: str = 'nearest',
		align_corners: bool = None,
		recompute_scale_factor: bool = None
	) -> None:
		super().__init__()
		self._size = size
		self._scale_factor = scale_factor
		self._mode = mode
		self._align_corners = align_corners
		self._recompute_scale_factor = recompute_scale_factor
		
	def forward(self, x: Tensor) -> Tensor:
		return interpolate(
			x,
			size=self._size,
			scale_factor=self._scale_factor,
			mode=self._mode,
			align_corners=self._align_corners
		)