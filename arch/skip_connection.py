from __future__ import annotations

from torch import nn, Tensor
from typing import Optional


class SkipMisuseException(Exception):
	pass


class SkipConnection(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self._storage: Optional[Tensor] = None
		self._save: bool = False
		self._done: bool = False

	def forward(self, x: Tensor):
		if self._done:
			raise SkipMisuseException
		if self._save:
			self._save = False
			self._storage = x
			return x
		else:
			self._done = True
			return x + self._storage