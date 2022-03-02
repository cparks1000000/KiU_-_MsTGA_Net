from typing import List

from torch import nn, Tensor

from KiUNet.modules.skip_connection import SkipConnection
from KiUNet.modules.util import create_base_downsample, create_base_upsample


# todo: this is a Unet without skips. Refactor this.
class autoencoder(nn.Module):
    def __init__(self, has_softmax) -> None:
        super().__init__()

        channels: List[int] = [64]
        for _ in range(4):
            channels.append(2*channels[-1])
        channels = [3] + channels

        modules: List[nn.Module] = []

        for a, b in zip(channels, channels[1:]):
            modules += create_base_downsample(a, b)

        channels.reverse()

        for a, b in zip(channels, channels[1:]):
            modules += create_base_upsample(a, b)

        if has_softmax:
            modules += nn.Softmax(dim=1)

        self._calculate = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self._calculate(x)


class Unet(nn.Module):
    def __init__(self, has_softmax: bool) -> None:
        super().__init__()

        modules: List[nn.Module] = []

        channels: List[int] = [32]
        for _ in range(4):
            channels.append(2 * channels[-1])
        channels = [3] + channels

        # A skip module records its first input and returns it. A skip module returns the sum of its second input and its record
        skips: List[nn.Module] = [SkipConnection() for _ in range(len(channels)-2) ]
        skips += [nn.Identity()]  # don't add a skip at the end

        for skip, a, b in zip(skips, channels, channels[1:] ):
            modules += [create_base_downsample(a, b), skip]

        skips.reverse()
        channels.reverse()

        for skip, a, b in zip(skips, channels, channels[1:]):
            modules += [create_base_upsample(a, b), skip]

        if has_softmax:
            modules.append( nn.Softmax(dim=1) )

        self._skips = skips
        self._calculate = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self._calculate(inputs)
        map(lambda x: x.reset(), self._skips)
        return output