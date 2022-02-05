from __future__ import annotations

from torch import nn, Tensor, cat
from typing import Optional, List


class SkipMisuseException(Exception):
	pass


class SkipConnection(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._storage: Optional[Tensor] = None
		self._save: bool = True
		self._dead: bool = False

	def forward(self, x: Tensor):
		if self._dead:
			raise SkipMisuseException
		if self._save:
			self._save = False
			self._storage = x
			return x
		else:
			self._dead = True
			return x + self._storage


class DenseSkip(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._storage: List[Tensor] = []

	def forward(self, x: Tensor) -> Tensor:
		self._storage.append(x)
		return cat(self._storage, dim=1)