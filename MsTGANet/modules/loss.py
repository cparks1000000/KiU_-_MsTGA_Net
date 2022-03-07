from functools import reduce

from torch import nn, Tensor


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

# todo: DiceLoss not callable
class MergerLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cross_entropy = nn.CrossEntropyLoss()
        self._dice = DiceLoss()

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        return self._cross_entropy(inputs, labels) + self._dice(inputs, labels)
