from __future__ import annotations
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from KiUNet.arch.mfrb import MFRB
from KiUNet.arch.parallel import Parallel, Split, Sum
from KiUNet.arch.skip_connection import SkipConnection
from KiUNet.arch.util import create_base_upsample, create_base_downsample, create_resnet_downsample, create_resnet_upsample, create_dense_downsample, create_dense_upsample
from KiUNet.arch.util import ModuleFactory


def skip_parallel() -> Parallel:
    return Parallel( SkipConnection(), SkipConnection() )


class BaseKiUNet(nn.Module):
    def __init__(self, channels_in: int,  has_softmax: bool, has_skips: bool, layer_function: ModuleFactory):
        super().__init__()

        layer_count = 3

        identity = Parallel( nn.Identity(), nn.Identity() )

        modules: List[nn.Module] = [ Split() ]

        channels: List[int] = [16]
        up_scales: List[Optional[float]] = [4]
        for _ in range(layer_count-1):
            channels.append(2 * channels[-1])
            up_scales += [up_scales[-1]*4]
        channels = [channels_in] + channels
        up_scales = [None] + up_scales

        # todo: Make this smarter
        # On its first use, a skip module records its input and returns it. On its second use, a skip module returns the sum of its input and its previous record. A skip module can not be used a third time.
        if has_skips:
            skips: List[Parallel] = [skip_parallel() for _ in range(len(channels) - 2)] + [identity]  # don't add a skip at the end
        else:
            skips: List[Parallel] = [identity for _ in range(len(channels) - 1)]

        for skip, a, b, up_scale in zip(skips, channels, channels[1:], up_scales[1:]):
            modules += [layer_function(skip, a, b, up_scale)]

        channels = [8] + channels[1:]
        channels.reverse()
        skips.reverse()
        up_scales.reverse()

        for skip, a, b, up_scale in zip(skips, channels, channels[1:], up_scales[1:]):
            modules += [layer_function(skip, a, b, up_scale)]

        modules += [
            Sum(),
            nn.Conv2d(channels[-1], 2, (1, 1)),
            nn.ReLU()
        ]

        if has_softmax:
            modules.append(nn.Softmax(dim=1))

        self._calculate = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self._calculate(x)


# skip is expected to be two parallel skip connections
def base_layer(skip: Parallel, channels_in: int, channels_out: int, up_scale: Optional[int]) -> nn.Module:
    last_layer: nn.Module = nn.Identity()
    if up_scale is not None:
        last_layer: MFRB = MFRB(channels_out,  up_scale)
    return nn.Sequential(
        Parallel(
            nn.Sequential( *create_base_downsample( channels_in, channels_out) ),
            nn.Sequential( *create_base_upsample( channels_in, channels_out) )
        ),
        skip,
        last_layer
    )


class KiUNet(BaseKiUNet):
    def __init__(self, channels_in: int,  has_softmax: bool, has_skips: bool):
        super().__init__(channels_in,  has_softmax, has_skips, base_layer)


def resnet_layer(skip: Parallel, channels_in: int, channels_out: int, up_scale: Optional[int]) -> nn.Module:
    last_layer: nn.Module = nn.Identity()
    if up_scale is not None:
        last_layer: MFRB = MFRB(channels_out,  up_scale)
    return nn.Sequential(
        Parallel(
            nn.Sequential( *create_resnet_downsample( channels_in, channels_out) ),
            nn.Sequential( *create_resnet_upsample( channels_in, channels_out) )
        ),
        skip,
        last_layer
    )


class ResKiUNet(BaseKiUNet):
    def __init__(self, channels_in: int,  has_softmax: bool, has_skips: bool):
        super().__init__(channels_in,  has_softmax, has_skips, resnet_layer)


def dense_layer(skip: Parallel, channels_in: int, channels_out: int, up_scale: Optional[int]) -> nn.Module:
    last_layer: nn.Module = nn.Identity()
    if up_scale is not None:
        last_layer: MFRB = MFRB(channels_out,  up_scale)
    return nn.Sequential(
        Parallel(
            nn.Sequential( *create_dense_downsample( channels_in, channels_out) ),
            nn.Sequential( *create_dense_upsample( channels_in, channels_out) )
        ),
        skip,
        last_layer
    )


class DenseKiUNet(BaseKiUNet):
    def __init__(self, channels_in: int,  has_softmax: bool, has_skips: bool):
        super().__init__(channels_in,  has_softmax, has_skips, dense_layer)


# todo: Refactor this.
class kiunet3d(nn.Module):

    def __init__(self, c=4, n=1,channels=128, groups = 16, norm='bn', num_classes=5):
        super(kiunet3d, self).__init__()

        # Entry flow
        self.encoder1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=1, bias=False)# H//2
        self.encoder2 = nn.Conv3d( n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.encoder3 = nn.Conv3d( 2*n, 4*n, kernel_size=3, padding=1, stride=1, bias=False)

        self.kencoder1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder2 = nn.Conv3d( n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kencoder3 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)

        self.downsample1 = nn.MaxPool3d(2, stride=2)
        self.downsample2 = nn.MaxPool3d(2, stride=2)
        self.downsample3 = nn.MaxPool3d(2, stride=2)
        self.kdownsample1 = nn.MaxPool3d(2, stride=2)
        self.kdownsample2 = nn.MaxPool3d(2, stride=2)
        self.kdownsample3 = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.kupsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.kupsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.kupsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2

        self.decoder1 = nn.Conv3d( 4*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.decoder2 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.decoder3 = nn.Conv3d( 2*n, c, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder1 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder2 = nn.Conv3d( 2*n, 2*n, kernel_size=3, padding=1, stride=1, bias=False)
        self.kdecoder3 = nn.Conv3d( 2*n, c, kernel_size=3, padding=1, stride=1, bias=False)

        self.intere1_1 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv3d(2*n,4*n,3, stride=1, padding=1)
        # self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv3d(4*n,2*n,3, stride=1, padding=1)
        # self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv3d(2*n,2*n,3, stride=1, padding=1)
        # self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv3d(n,n,3, stride=1, padding=1)
        # self.intd3_2bn = nn.BatchNorm2d(64)

        self.seg = nn.Conv3d(c, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))  #U-Net branch
        out1 = F.relu(F.interpolate(self.kencoder1(x),scale_factor=2,mode ='trilinear')) #Ki-Net branch
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere1_1(out1)),scale_factor=0.25,mode ='trilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere1_2(tmp)),scale_factor=4,mode ='trilinear')) #CRFB

        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        out1 = F.relu(F.interpolate(self.kencoder2(out1),scale_factor=2,mode ='trilinear'))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere2_1(out1)),scale_factor=0.0625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere2_2(tmp)),scale_factor=16,mode ='trilinear'))

        u2 = out
        o2 = out1

        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        out1 = F.relu(F.interpolate(self.kencoder3(out1),scale_factor=2,mode ='trilinear'))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intere3_1(out1)),scale_factor=0.015625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intere3_2(tmp)),scale_factor=64,mode ='trilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=2,mode ='trilinear'))  #U-NET
        out1 = F.relu(F.max_pool3d(self.kdecoder1(out1),2,2)) #Ki-NET
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.interd1_1(out1)),scale_factor=0.0625,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.interd1_2(tmp)),scale_factor=16,mode ='trilinear'))

        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=2,mode ='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder2(out1),2,2))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.interd2_1(out1)),scale_factor=0.25,mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.interd2_2(tmp)),scale_factor=4,mode ='trilinear'))

        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=2,mode ='trilinear'))
        out1 = F.relu(F.max_pool3d(self.kdecoder3(out1),2,2))

        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.seg(out))  #1*1 conv


        # out = self.soft(out)
        return out

