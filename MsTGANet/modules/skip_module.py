import torch
import torch.nn as nn
from torch import Tensor, cat

from MsTGANet.modules.convolutions import SkipConvolution, FeatureConvolution


# I don't think the middle_channels was that important, so I replaced it with total_channels_in // 4.
# That's what it was in their implementations.
class SkipModule(nn.Module):
    _saved: Tensor

    def __init__(self, decoder_channels_in: int, encoder_channels_in : int, channels_out: int, height: int, width: int):
        super().__init__()
        print("=============== MsGCS ===============")

        # Parameters have requires_grad=True as default.
        self.height_tensor: nn.Parameter = nn.Parameter(torch.randn([1, 1, height, 1]))
        self.width_tensor: nn.Parameter = nn.Parameter(torch.randn([1, 1, 1, width]))

        total_channels_in: int = decoder_channels_in + encoder_channels_in
        middle_channels: int = total_channels_in//4
        if middle_channels == 0:
            middle_channels = 1
        self._fusion = nn.Sequential(
            SkipConvolution(total_channels_in, middle_channels),
            SkipConvolution(middle_channels, 1)
        )
        self._activation: nn.Module = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self._normalize_channels: nn.Module = FeatureConvolution(total_channels_in, channels_out)

        self._uses: int = 0

    def forward(self, inputs: Tensor):
        self._uses += 1
        assert self._uses == 1 or self._uses == 2, "You tried to use a skip module twice."
        if self._uses == 1:
            self._saved = inputs
            return inputs

        else:
            combined_data = self._fusion(torch.cat([inputs, self._saved], dim=1))
            heat_map = self.height_tensor + self.width_tensor
            attention = self._activation(heat_map * combined_data)
            attended_encoder_data = attention * self._saved
            # changed: original concatenation did not concatenate the correct dimensions
            return self._normalize_channels(torch.cat([inputs, attended_encoder_data], dim=1))

    def get_saved(self) -> Tensor:
        assert self._uses == 1 or self._uses == 2, "You tried to use a skip module twice."
        assert self._uses == 1, "You haven't saved anything yet."
        return self._saved

    def reset(self) -> None:
        self._uses = 0


class SimpleSkipModule(SkipModule):
    """This is a MsGCS where the channels are all equal"""
    def __init__(self, channels: int, height: int, width: int):
        super().__init__(channels, channels, channels, height, width)

