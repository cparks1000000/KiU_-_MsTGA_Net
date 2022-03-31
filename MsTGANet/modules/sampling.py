from torch import nn as nn, Tensor

from MsTGANet.modules.convolutions import SamplingConvolution


class Sampling(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, scale_factor: float):
        assert (kernel_size-1) % 2 == 0, "The kernel size for Sampling should be odd."
        super().__init__()
        self._output = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
                SamplingConvolution(channels_in, channels_out, kernel_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._output(x)
