from itertools import product
from typing import List

import torch
from torch import nn as nn, Tensor


class AttentionBase(nn.Module):
    def __init__(self, channels_in: int, height: int, width: int, *,
                 max_dimensions: int = 128):
        super().__init__()

        # added initialization of channels_in so it can be accessed in forward
        self._channels_in = channels_in
        self._max_dimensions = max_dimensions
        
        self.height_tensor = nn.Parameter(torch.randn([1, channels_in // 8, height, 1]))
        self.width_tensor = nn.Parameter(torch.randn([1, channels_in // 8, 1, width]))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.query_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self.key_conv = nn.Conv2d(channels_in, channels_in // 8, 1)
        self.value_conv = nn.Conv2d(channels_in, channels_in, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, x_encoder: Tensor) -> Tensor:
        if x.size(2) * x.size(3) <= self._max_dimensions ** 2:
            return self._default_forward(x, x_encoder)
        return self._backup_forward(x, x_encoder)
    
    def _default_forward(self, x: Tensor, x_encoder: Tensor) -> Tensor:
        batch_size, channel_size, height, width = x.size()
        query = self.query_conv(x).view(batch_size, self._channels_in // 8, width * height).permute(0, 2, 1)

        key = self.key_conv(x_encoder).view(batch_size, self._channels_in // 8, width * height)

        energy_content = torch.bmm(query, key)

        content_position = (self.height_tensor + self.width_tensor).view(1, self._channels_in // 8, -1)
        content_position = torch.matmul(query, content_position)

        energy = energy_content+content_position
        attention = self.softmax(energy)
        value = self.value_conv(x_encoder).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel_size, width, height)

        return self.gamma * out + x
    
    def _backup_forward(self, x: Tensor, x_encoder: Tensor) -> Tensor:
        batch_size, channel_size, height, width = x.size()
        query_list: List[Tensor] = list(
            self.query_conv(x).view(batch_size, channel_size // 8, width * height).transpose(1, 2)
        )
        key_list: List[Tensor] = list(
            self.key_conv(x_encoder).view(batch_size, channel_size // 8, width * height)
        )
        value_list: List[Tensor] = list(
            self.value_conv(x_encoder).view(batch_size, channel_size, width * height)
        )
        heat = (self.height_tensor + self.width_tensor).view(1, self._channels_in // 8, -1)
        
        holder = torch.zeros(batch_size, channel_size, height*width)
        for b in range(batch_size):
            query = query_list[b]
            key = key_list[b]
            value = value_list[b]
            for c, i in product(range(channel_size), range(height*width)):
                holder[b, c, i] = value[c] @ torch.softmax(query[i] @ (key + heat), dim=0).transpose(0, 1)
        
        out = holder.view(batch_size, channel_size, width, height)

        return self.gamma * out + x