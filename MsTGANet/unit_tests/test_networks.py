from __future__ import annotations

from torchvision.datasets import SVHN, MNIST
from torchvision.transforms import Compose, ToTensor

from MsTGANet.modules.sampling_factory import DefaultUpsampleFactory, DefaultDownsampleFactory
from MsTGANet.networks.template_network import UNetwork, KiNetwork
from MsTGANet.options.base_options import BaseOptions
from MsTGANet.datasets.base_dataset import BaseDataset
from datasets.dataset_wrapper import DatasetWrapper
from datasets.load_dataset import LoadDataset

upsample_factory = DefaultUpsampleFactory()
downsample_factory = DefaultDownsampleFactory()

# train_set: BaseDataset = LoadDataset(
#             csv_file="/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/datasets/GLAS/Train_Set/Grade_Train.csv",
#             root_dir="/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/datasets/GLAS/Train_Set/",
#         )


# Used SVHN dataset, there is an issue with 28 x 28 images
train_set: BaseDataset = DatasetWrapper(SVHN(
    root="data", split='train', transform=ToTensor()
))

test_set: BaseDataset = DatasetWrapper(SVHN(
    root="data", split='test', transform=ToTensor()
))

# 3 channels for color images, each is 32x32, and we are segmenting into two classes
opt: BaseOptions = BaseOptions(3, 32, 32, 2)


def test_u_network() -> None:
    network = UNetwork(opt, test_set, train_set)
    network.do_train()


def test_ki_network() -> None:
    network = KiNetwork(opt, test_set, train_set)
    network.do_train()


if __name__ == "__main__":
    test_u_network()
