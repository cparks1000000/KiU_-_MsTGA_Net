from __future__ import annotations
from typing import List

from MsTGANet.models.base_model import BaseModel
from MsTGANet.models.template import Template
from MsTGANet.modules.sampling_factory import DefaultUpsampleFactory, DefaultDownsampleFactory
from MsTGANet.networks.merger_network import TemplateNetwork
from MsTGANet.options.base_options import BaseOptions
from MsTGANet.datasets.base_dataset import BaseDataset, LoadDataset


upsample_factory = DefaultUpsampleFactory()
downsample_factory = DefaultDownsampleFactory()


class TestUNet(Template, BaseModel):
    # noinspection PyDefaultArgument
    def __init__(self, channels_in: int, height: int, width: int, number_of_classes: int, *,
                 channels_list: List[int] = [32, 64, 128, 256, 512]):
        Template.__init__(
            self, channels_in, height, width, number_of_classes,
            encoder_sampling=downsample_factory,
            decoder_sampling=upsample_factory,
            channels_list=channels_list,
        )
        BaseModel.__init__(self, "test_u_net", channels_in, height, width, number_of_classes)


class TestUNetwork(TemplateNetwork):
    def __init__(self, opt: BaseOptions, test_set: BaseDataset, train_set: BaseDataset):
        model: BaseModel = TestUNet(opt.channels_in, opt.height, opt.width, opt.number_of_classes)
        super().__init__(opt, test_set, train_set, model)


train_data: BaseDataset = LoadDataset(
        csv_file="/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/datasets/GLAS/Train_Set/Grade_Train.csv",
        root_dir="GLAS/Train_Set/")

test_data: BaseDataset = LoadDataset(
        csv_file="/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/datasets/GLAS/Test_Set/Grade_Test.csv",
        root_dir="GLAS/Test_Set/")

model = TestUNet(BaseOptions.channels_in, BaseOptions.height, BaseOptions.width, BaseOptions.number_of_classes)


