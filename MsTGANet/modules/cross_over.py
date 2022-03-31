from __future__ import annotations

from torch import nn, Tensor

from MsTGANet.modules.sampling import Sampling


class CrossOver(nn.Module):
	def __init__(self, channels: int, up_scale: float ) -> None:
		super().__init__()
		self._downsample = Sampling(channels, channels, 3, 1/up_scale)
		self._upsample = Sampling(channels, channels, 3, up_scale)

	def forward(self, pair: (Tensor, Tensor)) -> (Tensor, Tensor):
		# changed: took in the pair of tensors as input, split them for computations, then returned another pair
		u_in = pair[0]
		k_in = pair[1]
		u_out: Tensor = u_in + self._downsample(k_in)
		k_out: Tensor = k_in + self._upsample(u_in)
		return u_out, k_out
