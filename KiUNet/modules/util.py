from typing import List

from torch import nn

from KiUNet.arch.interpolate import Interpolate
from KiUNet.arch.parallel import Split, Parallel, Sum
from KiUNet.arch.skip_connection import DenseSkip


def create_base_downsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        nn.Conv2d(channels_in, channels_out, (3, 3), padding=1),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
    ]
    return output


def create_base_upsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        nn.Conv2d(channels_in, channels_out, (3, 3), padding=1),
        Interpolate(scale_factor=(2.0, 2.0), mode='bilinear'),
        nn.ReLU()
    ]
    return output


def create_resnet_block(channels_in: int, channels_out: int) -> nn.Module:
    return nn.Sequential(
        Split(),
        Parallel(
            nn.Conv2d(channels_in, channels_out, (3, 3), padding=1),
            nn.Conv2d(channels_in, channels_out, (1, 1))
        ),
        Sum()
    )


def create_resnet_downsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        create_resnet_block(channels_in, channels_out),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
    ]
    return output


def create_resnet_upsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        create_resnet_block(channels_in, channels_out),
        Interpolate(scale_factor=(2.0, 2.0), mode='bilinear'),
        nn.ReLU()
    ]
    return output


def create_dense_downsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        nn.Conv2d(channels_in, channels_out, (3, 3), padding=1),
        DenseBlock(in_planes=channels_out),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
    ]
    return output


def create_dense_upsample(channels_in: int, channels_out: int) -> List[nn.Module]:
    output: List[nn.Module] = [
        nn.Conv2d(channels_in, channels_out, (3, 3), padding=1),
        DenseBlock(in_planes=channels_out),
        Interpolate(scale_factor=(2.0, 2.0), mode='bilinear'),
        nn.ReLU()
    ]
    return output


def R(x: List) -> list:
    return list(range(len(x)))


class DenseBlock(nn.Module):

    def __init__(self, in_planes):
        super().__init__()
        dense_skip = DenseSkip()
        batch_norm_1 = nn.BatchNorm2d(in_planes)
        batch_norm_2 = nn.BatchNorm2d(in_planes//4)
        planes: List[int] = [in_planes, in_planes + in_planes//4, in_planes + 2*(in_planes//4), in_planes + 3*(in_planes//4)]
        modules: List[nn.Module] = []
        for plane in planes:
            modules += [
                nn.Conv2d( plane, in_planes, kernel_size=(1, 1) ),
                batch_norm_1,
                nn.ReLU(),
                nn.Conv2d( in_planes, in_planes//4, kernel_size=(3, 3), padding=1 ),
                batch_norm_2,
                nn.ReLU(),
                dense_skip
            ]
        self._model = nn.Sequential(*modules)

    def forward(self, x):
        return x + self._model(x)