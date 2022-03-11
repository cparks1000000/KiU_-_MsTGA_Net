from typing import List

import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList

from MsTGANet.modules.sampling_factory import SamplingFactory
from modules.attention.attention_decoder import AttentionDecoder
from modules.attention.attention_encoder import AttentionEncoder


class Attention(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, channels_list: List[int], height: int, width: int, skip_modules: ModuleList, sampling_factory: SamplingFactory):
        print("============= MsTNL =============")
        super().__init__()
        self._skip_modules = skip_modules
        self._encoder = AttentionEncoder(channels_list[-1], channels_list[:-1], height, width, sampling_factory)
        self._decoder = AttentionDecoder(channels_list[-1], height, width)

    def forward(self, x: Tensor):
        x_list = self.get_saved(self._skip_modules)
        x_encoder = self._encoder(x, x_list)
        x_out = self._decoder(x, x_encoder)
        return x_out

    @classmethod
    def get_saved(cls, skips: ModuleList) -> List[Tensor]:
        output: List[Tensor] = []
        for skip in skips:
            output.append(skip.get_saved())
        return output