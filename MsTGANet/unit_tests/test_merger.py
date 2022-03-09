from __future__ import annotations
from typing import List

from MsTGANet.models.base_model import BaseModel
from MsTGANet.models.template import Template
from MsTGANet.models.merger import Merger
from MsTGANet.networks.merger_network import TemplateNetwork
from MsTGANet.options.base_options import BaseOptions
from MsTGANet.datasets.base_dataset import BaseDataset, LoadDataset


class TestMergerNetwork(TemplateNetwork):
    def __init__(self, opt: BaseOptions, test_set: BaseDataset, train_set: BaseDataset):
        model: BaseModel = Merger(opt.channels_in, opt.height, opt.width, opt.number_of_classes)
        super().__init__(opt, test_set, train_set, model)