from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn


class SamplingFactory(ABC):
	@abstractmethod
	def get(self) -> nn.Module: ...

	@abstractmethod
	def get_scale(self) -> float: ...

	@abstractmethod
	def scale(self, x: float) -> float: ...

	def scale_int(self, x: int) -> int:
		return int(self.scale(x))


class DefaultDownsampleFactory(SamplingFactory):
	def __init__(self, scale: int = 2):
		self._scale = scale

	def get(self) -> nn.Module:
		return nn.MaxPool2d(kernel_size=self._scale, stride=self._scale)

	def get_scale(self) -> float:
		return self._scale

	def scale(self, x: float) -> float:
		return x / self.get_scale()


class DefaultUpsampleFactory(SamplingFactory):
	def __init__(self, scale: int = 2):
		self._scale = scale

	def get(self) -> nn.Module:
		return nn.Upsample(scale_factor=self._scale)

	def get_scale(self) -> float:
		return self._scale

	def scale(self, x: float) -> float:
		return self.get_scale()*x
