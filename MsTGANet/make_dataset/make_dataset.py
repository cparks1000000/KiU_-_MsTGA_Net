from itertools import product

import torch
from torch import zeros_like, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import Pad, ToTensor
from torchvision.io import write_png


def convert(x: Tensor) -> Tensor:
    # noinspection PyTypeChecker
    return torch.round(255 * x).to(torch.uint8)


def generate(dataset: Dataset, name: str):
    dataloader = DataLoader(dataset)
    pad: Pad = Pad(2)
    
    for index, (image, _) in enumerate(dataloader):
        image = image.squeeze(0)
        new_image = pad(image)
        label = zeros_like(new_image)
        dim = len(new_image[0])
        for i, j in product(range(dim), range(dim)):
            if new_image[0, i, j].item() != 0:
                label[0, i, j] = 1
        write_png(convert(new_image), "./" + name + "/images/" + str(index) + ".png")
        write_png(convert(label), "./" + name + "/labels/" + str(index) + ".png")


if __name__ == "__main__":
    generate(
        datasets.MNIST('./', download=True, transform=ToTensor()),
        "train_set"
    )
    print("train set done")
    generate(
        datasets.MNIST('./', download=True, transform=ToTensor(), train=False),
        "test_set"
    )
    print("test set done")