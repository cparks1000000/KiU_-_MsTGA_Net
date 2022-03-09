from typing import List

from MsTGANet.models.base_model import BaseModel
from MsTGANet.models.template import Template
from MsTGANet.modules.sampling_factory import DefaultUpsampleFactory, DefaultDownsampleFactory

upsample_factory = DefaultUpsampleFactory()
downsample_factory = DefaultDownsampleFactory()


class KiNet(Template, BaseModel):
    # noinspection PyDefaultArgument
    def __init__(self, channels_in: int, height: int, width: int, number_of_classes: int, *,
                 channels_list: List[int] = [32, 64, 128, 256, 512]):
        Template.__init__(
            self, channels_in, height, width, number_of_classes,
            encoder_sampling=upsample_factory,
            decoder_sampling=downsample_factory,
            channels_list=channels_list,
        )
        BaseModel("ki_net", channels_in, height, width, number_of_classes)


class UNet(Template, BaseModel):
    # noinspection PyDefaultArgument
    def __init__(self, channels_in: int, height: int, width: int, number_of_classes: int, *,
                 channels_list: List[int] = [32, 64, 128, 256, 512]):
        Template.__init__(
            self, channels_in, height, width, number_of_classes,
            encoder_sampling=downsample_factory,
            decoder_sampling=upsample_factory,
            channels_list=channels_list,
        )
        BaseModel("u_net", channels_in, height, width, number_of_classes)
