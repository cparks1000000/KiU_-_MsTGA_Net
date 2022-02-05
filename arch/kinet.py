from typing import List

from torch import Tensor, nn

from arch.kiunet import create_base_upsample, create_base_downsample
from arch.skip_connection import SkipConnection


class kinetwithsk(nn.Module):

    def __init__(self, has_softmax: bool, has_skips: bool) -> None:
        super().__init__()

        modules: List[nn.Module] = []

        channels: List[int] = [32]
        for _ in range(2):
            channels.append(2 * channels[-1])
        channels = [1] + channels

        # todo: Make this smarter
        # A skip module records its first input and returns it. A skip module returns the sum of its second input and its record
        if has_skips:
            skips: List[nn.Module] = [SkipConnection() for _ in range(len(channels) - 2)] + [nn.Identity]  # don't add a skip at the end
        else:
            skips: List[nn.Module] = [nn.Identity() for _ in range(len(channels) - 1)]

        for skip, a, b in zip(skips, channels, channels[1:]):
            modules += [create_base_upsample(a, b), skip]

        skips.reverse()
        channels.reverse()

        for skip, a, b in zip(skips, channels, channels[1:]):
            modules += [create_base_downsample(a, b), skip]

        if has_softmax:
            modules.append(nn.Softmax(dim=1))

        self._calculate = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self._calculate(x)


class kitenet(kinetwithsk):
    def __init__(self, has_softmax: bool):
        super().__init__(has_softmax, False)