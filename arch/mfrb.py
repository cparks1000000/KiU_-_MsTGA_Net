from torch import \
	nn, \
	Tensor

from arch.interpolate import Interpolate


class MFRB(nn.Module):
	def __init__(self, channels: int, up_scale: float ) -> None:
		super().__init__()
		self._downsample = nn.Sequential(
				nn.Conv2d(channels, channels, 3, padding=1),
				nn.BatchNorm2d(channels),
				nn.ReLU(),
				Interpolate(scale_factor=(1/up_scale, 1/up_scale), mode='bilinear')
		)
		self._upsample = nn.Sequential(
				nn.Conv2d(channels, channels, 3, padding=1),
				nn.BatchNorm2d(channels),
				nn.ReLU(),
				Interpolate(scale_factor=(up_scale, up_scale), mode='bilinear')
		)

	def forward(self, u_in: Tensor, k_in: Tensor) -> (Tensor, Tensor):
		u_out: Tensor = u_in + self._downsample(k_in)
		k_out: Tensor = k_in + self._upsample(u_in)
		return u_out, k_out