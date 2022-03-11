import \
    torch
from torch import nn as nn, Tensor


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