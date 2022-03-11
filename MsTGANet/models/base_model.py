from __future__ import annotations

from abc import ABC
from typing import Any, Optional

import \
    torch
from bidict import bidict
from torch import nn


def build_signature(*args: Any) -> str:
    output = ""
    for arg in args[0:-1]:
        output += str(arg) + 'x'
    output += str(args[-1])
    return output


class ActivateGrad:
    def __init__(self, model: BaseModel) -> None:
        self._model = model

    def __enter__(self) -> None:
        self._model.set_requires_grad(True)

    # noinspection PyMissingTypeHints
    def __exit__(self, exc_type, exc_value, traceback):
        self._model.set_requires_grad(False)


class BaseModel(nn.Module, ABC):
    _default_requires_grad: bool = False
    _default_root: str = "_network_weights"
    _file_name_dictionary: bidict[str, str] = {}

    @classmethod
    def set_default_requires_grad(cls, flag: bool):
        cls._default_requires_grad = flag

    # The file_name parameter must be a property of the inheriting class.
    def __init__(self, file_name: str, *signature_args: Any):
        super().__init__()
        self._signature = build_signature(signature_args)
        assert file_name not in self._file_name_dictionary, "The file name " + file_name + " has been used twice."
        self._file_name_dictionary[file_name] = file_name
        self._file_name = file_name

    def save(self,  file_name: Optional[str] = None, root: str = _default_root) -> None:
        path = self.get_path(file_name, root)
        torch.save(self.state_dict(), path)

    def load(self, file_name: Optional[str] = None, root: str = _default_root) -> BaseModel:
        path = self.get_path(file_name, root)
        self.load_state_dict(torch.load(path))
        return self

    def set_requires_grad(self, flag: bool) -> BaseModel:
        for parameter in self.parameters():
            parameter.requires_grad = flag
        return self

    def activate_grad(self) -> ActivateGrad:
        return ActivateGrad(self)

    def get_path(self, file_name: Optional[str] = None, root: str = _default_root):
        if file_name is None:
            file_name = self._file_name
        return root + "/" + file_name + self._signature + ".pth"
