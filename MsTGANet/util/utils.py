from __future__ import annotations

from math import floor
from typing import Any, Union

import numpy
from numpy import ndarray
from torch import Tensor, nn
import torch
import numpy as np


def smooth_max(a: Union[Tensor, float], b: Union[Tensor, float], alpha: float):
    assert alpha > 0, "If alpha <= 0, the function is not smooth."
    return (a+b + torch.sqrt((a-b)**2 + alpha))/2


def build_string(*args: Any) -> str:
    output = ""
    for arg in args:
        output += str(arg)
    return output

def build_signature(*args: Any) -> str:
    output = ""
    if len(args) != 0:
        for arg in args[0:-1]:
            output += str(arg) + 'x'
        output += str(args[-1])
    return output


def update_convolution(inputs: int, kernel_size: int, stride: int, padding: int, dilation: int = 1):
    return floor((inputs+2*padding-dilation*(kernel_size-1)-1)/stride + 1)


def tensor_to_image(tensor: Tensor) -> ndarray:
    image = 127.5*(to_numpy(tensor[0]) + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def image_to_tensor(image: Tensor):
    return 2*image - 1


class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int) -> None:
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the work session ends!"
        self.n_epochs: int = n_epochs
        self.offset: int = offset
        self.decay_start_epoch: int = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# noinspection PyProtectedMember
def weights_init_normal(model: nn.Module):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.constant(model.bias, 0.0)


def to_numpy(input_tensor: Tensor) -> numpy.ndarray:
    return input_tensor.float().detach().cpu().numpy()


def copy_module_list(input_list: nn.ModuleList) -> nn.ModuleList:
    output_list = nn.ModuleList()
    for element in input_list:
        output_list.append(element)
    return output_list