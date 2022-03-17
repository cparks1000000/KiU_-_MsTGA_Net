# todo: Do we need this file? Could put these into utils in util folder

from typing import Any

import torch
from torch import nn, Tensor


class Log(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)


def may_print(flag: bool, *args: Any) -> None:
    if flag:
        print(*args)