from __future__ import annotations

from torchvision.transforms import Compose, ToTensor

from MsTGANet.modules.sampling_factory import DefaultUpsampleFactory, DefaultDownsampleFactory
from MsTGANet.networks.template_network import UNetwork, KiNetwork, MergerNetwork
from MsTGANet.options.base_options import BaseOptions
from MsTGANet.datasets.base_dataset import BaseDataset
from datasets.dataset_wrapper import DatasetWrapper
from datasets.load_dataset import LoadDataset

upsample_factory = DefaultUpsampleFactory()
downsample_factory = DefaultDownsampleFactory()


# to load custom dataset, input paths to where the image folder and labels folder are located
# transforms ToTensor, not sure if necessary
train_set: BaseDataset = LoadDataset('/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/unit_tests/data/dataset/train_set/images',
                                     '/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/unit_tests/data/dataset/train_set/labels',
                                     transform=ToTensor(), label_transform=ToTensor())

# to load custom dataset, input paths to where the image folder and labels folder are located
# transforms ToTensor, not sure if necessary
test_set: BaseDataset = LoadDataset('/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/unit_tests/data/dataset/test_set/images',
                                     '/Users/juliabrixey/Desktop/Research/KiUNet-MsTGANet/MsTGANet/unit_tests/data/dataset/test_set/labels',
                                     transform=ToTensor(), label_transform=ToTensor())

opt: BaseOptions = BaseOptions(1, 32, 32, 2)


def test_u_network() -> None:
    network = UNetwork(opt, test_set, train_set)
    network.do_train()


def test_ki_network() -> None:
    network = KiNetwork(opt, test_set, train_set)
    network.do_train()


def test_merger() -> None:
    network = MergerNetwork(opt, test_set, train_set)
    network.do_train()


if __name__ == "__main__":
    test_merger()
