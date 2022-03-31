from functools import reduce

from numpy import ones_like
from torch import nn, Tensor, torch


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._softmax = nn.Softmax(dim=1)
        self._nll = nn.NLLLoss(reduction='sum')

    # noinspection PyMethodMayBeStatic
    def forward(self, inputs: Tensor, labels: Tensor, smoothing_factor: float = 1) -> Tensor:
        inputs = self._softmax(inputs)
        entries_in_labels = reduce(lambda x, y: x*y, labels.size())

        intersection = self._nll(inputs, labels).sum()
        dice = (2. * intersection + smoothing_factor) / (inputs.sum() + entries_in_labels + smoothing_factor)

        return 1 - dice


class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._loss: nn.Module = nn.NLLLoss()
        self._softmax = nn.Softmax(dim=1)

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        # For some reason, NLLLoss has a negative sign...
        return -self.negative_cropped_log(-self._loss(self._softmax(prediction), label))

    # In general, any continuous function, defined piecewise, can be rewritten to be
    # defined in terms of logarithms.
    def negative_cropped_log(self, inputs: Tensor) -> Tensor:
        """<latex>Returns $-ln(x)$ if $x > e^{-10}$ and $11-x/e^{-10}$ otherwise.</latex>"""
        # noinspection PyTypeChecker
        cutoff = torch.exp(-10*ones_like(inputs))
        lower = torch.min(cutoff, inputs)
        upper = torch.max(cutoff, inputs)
        return 1-lower/cutoff - torch.log(upper)


class MergerLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # changed: switched CrossEntropyLoss() with CELoss()
        self._cross_entropy = CELoss()
        self._dice = DiceLoss()

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return self._cross_entropy(inputs, labels) + self._dice(inputs, labels)
