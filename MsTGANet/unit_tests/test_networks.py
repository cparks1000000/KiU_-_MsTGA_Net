from __future__ import annotations

from torchvision.datasets import MNIST

from MsTGANet.modules.sampling_factory import DefaultUpsampleFactory, DefaultDownsampleFactory
from MsTGANet.networks.template_network import UNetwork
from MsTGANet.options.base_options import BaseOptions
from MsTGANet.datasets.base_dataset import BaseDataset
from datasets.dataset_wrapper import DatasetWrapper
from datasets.load_dataset import LoadDataset

upsample_factory = DefaultUpsampleFactory()
downsample_factory = DefaultDownsampleFactory()

train_set: BaseDataset = LoadDataset(
            csv_file="datasets/GLAS/Train_Set/Grade_Train.csv",
            root_dir="datasets/GLAS/Train_Set/")

test_set: BaseDataset = DatasetWrapper(MNIST(

))
opt: BaseOptions = BaseOptions(3, 256, 256, 2)


def test_u_network() -> None:
    network = UNetwork(opt, test_set, train_set)
    network.do_train()


if __name__ == "__main__":
    test_u_network()
