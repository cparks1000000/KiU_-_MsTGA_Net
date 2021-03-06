from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor

from MsTGANet.models.base_model import BaseModel
from MsTGANet.modules.convolutions import ConvolutionBlock
from MsTGANet.modules.attention.attention import Attention
from MsTGANet.modules.sampling_factory import SamplingFactory, DefaultDownsampleFactory, DefaultUpsampleFactory
from MsTGANet.modules.skip_module import SimpleSkipModule
from MsTGANet.util.utils import copy_module_list, reverse


class Template(BaseModel):
    # noinspection PyDefaultArgument
    def __init__(self,
                 name: str,
                 channels_in: int,
                 height: int,
                 width: int,
                 number_of_classes: int,
                 encoder_sampling: SamplingFactory = DefaultDownsampleFactory(),
                 decoder_sampling: SamplingFactory = DefaultUpsampleFactory(),
                 *,
                 channels_list: List[int] = [32, 64, 128, 256, 512]) -> None:
        super().__init__(name)
        print("================ MsTGANet ================")

        # changed: implemented initial_block in forward
        self.initial_block = ConvolutionBlock(channels_in, channels_list[0])

        self.skip_modules: nn.ModuleList = nn.ModuleList()
        self.encoder_blocks: nn.ModuleList = nn.ModuleList()

        for channels_in, channels_out in zip(channels_list, channels_list[1:]):
            self.skip_modules.append(SimpleSkipModule(channels_in, width, height))
            self.encoder_blocks.append(nn.Sequential(
                    encoder_sampling.get(),
                    ConvolutionBlock(channels_in, channels_out)
            ))
            # todo: Can we work around this problem?
            # 28 x 28
            assert encoder_sampling.scale_int(width) == encoder_sampling.scale(width), "This resolution is not supported yet."
            width = encoder_sampling.scale_int(width)
            assert encoder_sampling.scale_int(height) == encoder_sampling.scale(height), "This resolution is not supported yet."
            height = encoder_sampling.scale_int(height)

        self.transformer = Attention(channels_list, height, width, copy_module_list(self.skip_modules), encoder_sampling)
        channels_list.reverse()
        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        for channels_in, channels_out in zip(channels_list, channels_list[1:]):
            self.decoder_blocks.append(nn.Sequential(
                    decoder_sampling.get(),
                    ConvolutionBlock(channels_in, channels_out)
            ))
            # todo: Can we work around this problem?
            # 28 x 28
            assert decoder_sampling.scale_int(width) == decoder_sampling.scale(width), "This resolution is not supported yet."
            width = decoder_sampling.scale_int(width)
            assert decoder_sampling.scale_int(height) == decoder_sampling.scale(height), "This resolution is not supported yet."
            height = decoder_sampling.scale_int(height)

        # changed: reversed channels list back to original state
        channels_list.reverse()
        self.final_block = nn.Sequential(
            nn.Conv2d(channels_list[0], number_of_classes, kernel_size=1),
            nn.Sigmoid()
        )

    # We won't use the forward directly.
    # Forward output is given to respective classes
    def forward(self, x: Tensor) -> Tensor:
        layer: nn.Module

        # changed: added passing input through initial block to get correct number of channels
        encoded: Tensor = self.initial_block(x)

        for skip, layer in zip(self.skip_modules, self.encoder_blocks):
            encoded = layer(skip(encoded))

        transformed: Tensor = self.transformer(encoded)

        # changed: created module list reverse in utils
        self.skip_modules = reverse(self.skip_modules)
        decoded = transformed

        for layer, skip in zip(self.decoder_blocks, self.skip_modules):
            decoded = skip(layer(decoded))

        self.skip_modules = reverse(self.skip_modules)
        map(lambda skip_module: skip_module.reset(), self.skip_modules)

        return self.final_block(decoded)
