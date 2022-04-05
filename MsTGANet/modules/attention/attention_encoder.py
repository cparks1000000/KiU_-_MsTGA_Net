from typing import List

from torch import nn as nn, Tensor, zeros_like

from modules.attention.attention_base import AttentionBase
from modules.sampling import Sampling
from modules.sampling_factory import SamplingFactory


class AttentionEncoder(AttentionBase):
    # noinspection PyDefaultArgument
    def __init__(self, channels_in: int, filters: List[int], height: int, width: int, sampling_factory: SamplingFactory, *,
                 max_dimensions: int = 128):
        super().__init__(channels_in, height, width, max_dimensions=max_dimensions)

        scales: List[float] = [1]
        for _ in filters:
            scales.append(sampling_factory.scale(scales[-1]))

        scales = scales[1:]
        scales.reverse()

        self._downsamples: nn.ModuleList = nn.ModuleList()
        # noinspection PyShadowingBuiltins
        for filter, scale in zip(filters, scales):
            self._downsamples.append(
                    Sampling(filter, channels_in, 3, scale)
            )

    def forward(self, final_encoded: Tensor, encoded_list: List[Tensor]) -> Tensor:
        total: Tensor = zeros_like(final_encoded)
        for encoded_data, down_sample in zip(encoded_list, self._downsamples):
            total = total + down_sample(encoded_data)
        return super().forward(final_encoded, total)
