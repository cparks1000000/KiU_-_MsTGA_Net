from __future__ import annotations

from typing import List

from torch import nn, Tensor

from MsTGANet.models.KiNet_and_UNet import KiNet, UNet
from MsTGANet.models.base_model import BaseModel
from MsTGANet.modules.cross_over import CrossOver
from MsTGANet.modules.parallel import Parallel, Split, Sum
from MsTGANet.modules.sampling_factory import DefaultDownsampleFactory, DefaultUpsampleFactory


def form_parallel_list(left_list: nn.ModuleList, right_list: nn.ModuleList) -> nn.ModuleList:
	output: nn.ModuleList = nn.ModuleList()
	for left, right in zip(left_list, right_list):
		output.append(Parallel(left, right))
	return output


class Merger(BaseModel):
	# noinspection PyDefaultArgument
	def __init__(self, channels_in: int, height: int, width: int, number_of_classes: int, *,
				 channels_list: List[int] = [32, 64, 128, 256, 512]) -> None:
		super().__init__("merger", channels_in, height, width, number_of_classes)
		upsample_factory = DefaultUpsampleFactory()
		downsample_factory = DefaultDownsampleFactory()
		assert upsample_factory.get_scale() == downsample_factory.get_scale(), "The upsample and downsample scales must agree."
		upscales: List[float] = [1]
		for _ in channels_list[2:]:
			upscales.append(upsample_factory.scale(upscales[-1]))

		u = UNet(channels_in, height, width, number_of_classes, channels_list=channels_list)

		k = KiNet(channels_in, height, width, number_of_classes, channels_list=channels_list)

		self.skips: nn.ModuleList = form_parallel_list(u.skip_modules, k.skip_modules)
		self.split: Split = Split()

		self.initial: Parallel = Parallel(u.initial_block, k.initial_block)
		self.encoders: nn.ModuleList = form_parallel_list(u.encoder_blocks, k.decoder_blocks)
		self.transformer: Parallel = Parallel(u.transformer, k.transformer)
		self.decoders: nn.ModuleList = form_parallel_list(u.decoder_blocks, k.decoder_blocks)
		self.final: Parallel = Parallel(u.final_block, k.final_block)
		self.sum: Sum = Sum()

		self.encoder_crossing: nn.ModuleList = nn.ModuleList()
		self.decoder_crossing: nn.ModuleList = nn.ModuleList()
		for channels, upscale in zip(channels_list, upscales):
			self.encoder_crossing.append(CrossOver(channels, upscale))
			self.decoder_crossing.append(CrossOver(channels, upscale))
		# noinspection PyTypeChecker
		self.decoder_crossing: nn.ModuleList = self.decoder_crossing[::-1]

	def forward(self, x: Tensor) -> Tensor:
		x = self.split(x)
		x = self.initial(x)
		for skip, cross_over, block in zip(self.skips, self.encoder_crossing, self.encoders):
			x = skip(x)
			x = cross_over(x)
			x = block(x)
		x = self.transformer(x)
		for block, cross_over, skip in zip(self.decoders, self.decoder_crossing, self.skips):
			x = block(x)
			x = cross_over(x)
			x = skip(x)
		x = self.final(x)
		x = self.sum(x)
		return x