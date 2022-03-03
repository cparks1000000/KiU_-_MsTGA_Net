from typing import List

import torch
import torch.nn as nn
from torch import Tensor, zeros_like

from MsTGANet.modules.sampling import Sampling
from MsTGANet.modules.sampling_factory import SamplingFactory
from MsTGANet.modules.skip_module import SkipModule


class AttentionEncoder(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, channels: int, filters: List[int], height: int, width: int, sampling_factory: SamplingFactory):
        super().__init__()
        print("================= Multi_Head_Encoder =================")

        # Parameters have requires_grad=True as default.
        self.height_tensor = nn.Parameter(torch.randn([1, channels // 8, height, 1]))
        self.width_tensor = nn.Parameter(torch.randn([1, channels // 8, 1, width]))
        self.gamma = nn.Parameter(torch.zeros(1))

        scales: List[float] = [1]
        for _ in filters:
            scales.append(sampling_factory.scale(scales[-1]))

        scales = scales[1:]
        scales.reverse()

        self._downsamples: nn.ModuleList = nn.ModuleList()
        # noinspection PyShadowingBuiltins
        for filter, scale in zip(filters, scales):
            self._downsamples.append(
                    Sampling(filter, channels, 3, scale)
            )

        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, final_encoded: Tensor, encoded_list: List[Tensor]) -> Tensor:
        batch_size, C, width, height = final_encoded.size()

        total: Tensor = zeros_like(final_encoded)
        for encoded_data, down_sample in zip(encoded_list, self._downsamples):
            total = total + down_sample(encoded_data)

        proj_query = self.query_conv(total).view(batch_size, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(final_encoded).view(batch_size, -1, width * height)

        energy_content = torch.bmm(proj_query, proj_key)

        content_position = (self.height_tensor + self.width_tensor).view(1, self.chanel_in // 8, -1)
        content_position = torch.matmul(proj_query, content_position)
        energy = energy_content + content_position
        attention = self.softmax(energy)
        proj_value = self.value_conv(final_encoded).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + final_encoded
        return out


class AttentionDecoder(nn.Module):
    def __init__(self, channels_in: int, height: int, width: int):
        super().__init__()
        print("================= Multi_Head_Decoder =================")

        self.height_tensor = nn.Parameter(torch.randn([1, channels_in // 8, height, 1]))
        self.width_tensor = nn.Parameter(torch.randn([1, channels_in // 8, 1, width]))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.query_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self.key_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self.value_conv = nn.Conv2d(channels_in, channels_in, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, x_encoder: Tensor) -> Tensor:
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x_encoder).view(batch_size, -1, width * height)

        energy_content = torch.bmm(proj_query, proj_key)

        content_position = (self.height_tensor + self.width_tensor).view(1, self.channels_in // 8, -1)
        content_position = torch.matmul(proj_query, content_position)

        energy = energy_content+content_position
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_encoder).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class Attention(nn.Module):
    # noinspection PyDefaultArgument
    def __init__(self, channels_list: List[int], height: int, width: int, skip_modules: List[SkipModule], sampling_factory: SamplingFactory):
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
    def get_saved(cls, skips: List[SkipModule]) -> List[Tensor]:
        output: List[Tensor] = []
        for skip in skips:
            output.append(skip.get_saved())
        return output