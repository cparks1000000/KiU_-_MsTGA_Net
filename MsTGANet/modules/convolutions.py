from __future__ import annotations

from torch import nn, Tensor


# todo: kernel_size, stride, padding in opt?
class ConvolutionModule(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        assert stride > 0, "Stride must be at least one."
        assert padding >= 0, "Padding must be at least zero."
        assert kernel_size > 0, "Kernel size must be at least one."
        # noinspection PyArgumentEqualDefault
        self._modules: nn.Module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._modules(x)


class FeatureConvolution(ConvolutionModule):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__(channels_in, channels_out, 3, 1, 1)


class SkipConvolution(ConvolutionModule):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__(channels_in, channels_out, 1, 1, 0)


class SamplingConvolution(ConvolutionModule):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int) -> None:
        assert (kernel_size-1) % 2 == 0, "The kernel size for Sampling should be odd."
        super().__init__(channels_in, channels_out, kernel_size, padding=kernel_size//2, stride=1)


class ConvolutionBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # noinspection PyArgumentEqualDefault
        self._modules: nn.Module = nn.Sequential(
            FeatureConvolution(channels_in, channels_out),
            FeatureConvolution(channels_out, channels_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._modules(x)
