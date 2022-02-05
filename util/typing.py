from __future__ import annotations

from typing import NewType, Callable

from torch import nn

ModuleFactory = NewType('ModuleFactory', Callable[[], nn.Module])