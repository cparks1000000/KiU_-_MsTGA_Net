import torch
import torch.nn as nn
from torch import Tensor


class MsGCS(nn.Module):
    def __init__(self, decoder_channels_in: int, encoder_channels_in: int, channels_out: int, height: int, width: int):
        super(MsGCS, self).__init__()
        print("=============== MsGCS ===============")

        total_channels_in: int = decoder_channels_in + encoder_channels_in
        self.Conv = nn.Sequential(
            nn.Conv2d(total_channels_in, channels_out, kernel_size=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.height_tensor = nn.Parameter(torch.randn([1, 1, height, 1]), requires_grad=True)
        self.width_tensor = nn.Parameter(torch.randn([1, 1, 1, width]), requires_grad=True)
        self.bn = nn.BatchNorm2d(1)
        self.activation = nn.Sigmoid()

    def forward(self, decoder_data: Tensor, encoder_data: Tensor):
        x_Multi_Scale = self.Conv(torch.cat([decoder_data, encoder_data], dim=1))
        content_position = self.height_tensor + self.width_tensor
        x_att_multi_scale = self.activation(self.bn(content_position * x_Multi_Scale))

        return x_att_multi_scale * encoder_data
