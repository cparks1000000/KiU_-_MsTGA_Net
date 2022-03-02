from __future__ import annotations

from torch import nn, Tensor, cat
from typing import Optional, List


class SkipMisuseException(Exception):
	pass


class SkipConnection(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._storage: Optional[Tensor] = None
		self._uses: int = 0

	def forward(self, x: Tensor):
		self._uses += 1
		if self._uses == 1:
			self._storage = x
			return x
		if self._uses == 2:
			return x + self._storage
		raise SkipMisuseException

	def reset(self) -> None:
		self._uses = 0


class DenseSkip(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._storage: List[Tensor] = []

	def forward(self, x: Tensor) -> Tensor:
		self._storage.append(x)
		return cat(self._storage, dim=1)