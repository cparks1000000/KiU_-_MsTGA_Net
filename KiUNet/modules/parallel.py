from __future__ import annotations

from torch import nn, Tensor


class Parallel(nn.Module):
	def __init__(self, left_module: nn.Module, right_module: nn.Module) -> None:
		super().__init__()
		self._right: nn.Module = right_module
		self._left: nn.Module = left_module

	def forward(self, pair: (Tensor, Tensor)) -> (Tensor, Tensor):
		return self._right(pair[0]), self._left(pair[1])


class Split(nn.Module):
	# noinspection PyMethodMayBeStatic
	def forward(self, x: Tensor) -> (Tensor, Tensor):
		return x, x


class Sum(nn.Module):
	# noinspection PyMethodMayBeStatic
	def forward(self, x: (Tensor, Tensor) ):
		return x[0] + x[1]