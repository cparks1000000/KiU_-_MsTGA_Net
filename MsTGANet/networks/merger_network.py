from __future__ import annotations

from typing import List, Callable

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ..datasets.base_dataset import BaseDataset
from ..models.KiNet_and_UNet import UNet
from ..models.base_model import BaseModel
from ..models.merger import Merger

from ..modules.loss import MergerLoss
from ..modules.util import may_print
from ..options.base_options import BaseOptions


class TemplateNetwork(nn.Module):
    def __init__(self, opt: BaseOptions, test_set: BaseDataset, train_set: BaseDataset, model: BaseModel):
        super().__init__()
        self._model: BaseModel = model
        self._loss_function: nn.Module = MergerLoss()
        self._optimizer: torch.optim.Optimizer = Adam(self._module.parameters(), opt.learning_rate)
        self._scheduler: StepLR = StepLR(self._optimizer, opt.epoch_between_decay, opt.decay_rate)

        self._train_loader: DataLoader = DataLoader(
                train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.dataloader_threads
        )
        # noinspection PyArgumentEqualDefault
        self._test_loader: DataLoader = DataLoader(
                test_set, batch_size=1, shuffle=True, num_workers=opt.dataloader_threads
        )
        self._opt = opt

    def do_train(self) -> None:
        for epoch_number in range(1, self._opt.number_of_epochs + 1):
            self._do_epoch(epoch_number)
        self._model.save()

    def _do_epoch(self, epoch_number: int) -> float:
        opt = self._opt
        total_loss: float = 0
        for batch_number, batch in enumerate(self._train_loader, 1):
            images = batch[0].to(opt.device)
            labels = batch[1].to(opt.device)
            total_loss += self._do_batch(images, labels, epoch_number, batch_number)
        self._scheduler.step()
        may_print(opt.verbose, "The loss for epoch", epoch_number, "was", str(total_loss) + ".")
        return total_loss

    def _do_batch(self, images: Tensor, labels: Tensor, epoch_number: int, batch_number: int) -> float:
        self._optimizer.zero_grad()
        output: Tensor = self._model(images)
        loss: Tensor = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()
        may_print(self._opt.verbose and batch_number % 10 == 0,
                  "The loss for batch", batch_number, "of epoch", epoch_number, "was", str(loss)+".")
        return loss.item()

    # todo: Calculate the Ppmcc from MsTGANet?
    def do_test(self, load: bool = False):
        if load:
            self._model.load()

        true_positives: List[int] = []
        false_positives: List[int] = []
        false_negatives: List[int] = []
        for batch in self._test_loader:
            image = batch[0].to(self._opt.device)
            label = batch[0].to(self._opt.device)
            probability = self._model(image)
            true_positives.append( self.true_positive(probability, label) )
            false_positives.append( self.false_positives(probability, label) )
            false_negatives.append( self.false_negative(probability, label) )

        IoU: float = sum(true_positives)/sum(true_positives + false_positives + false_negatives)
        dice: float = 0
        for true_positive, false_positive, false_negative in zip(true_positives, false_positives, false_negatives):
            dice += 2*true_positive/(2*true_positive + false_positive + false_negative)
        pearson: float = sum(true_positives)/sum(true_positives + false_positives)

        return IoU, dice, pearson

    def true_positive(self, probability: Tensor, label: Tensor) -> int:
        def condition(probability_pixel: int, label_pixel: int):
            return probability_pixel == label_pixel
        return self._test_help(probability, label, condition)

    def false_negative(self, probability: Tensor, label: Tensor) -> int:
        def condition(probability_pixel: int, label_pixel: int):
            return probability_pixel == self._opt.background_label and label_pixel != self._opt.background_label
        return self._test_help(probability, label, condition)

    def false_positive(self, probability: Tensor, label: Tensor) -> int:
        def condition(probability_pixel: int, label_pixel: int):
            return probability_pixel != self._opt.background_label and label_pixel != probability_pixel
        return self._test_help(probability, label, condition)

    # def true_negative(self, probability: Tensor, label: Tensor) -> int:
    #     def condition(probability_pixel: int, label_pixel: int):
    #         return probability_pixel == self._opt.background_label and label_pixel == self._opt.background_label
    #     return self._test_help(probability, label, condition)

    def _test_help(self, probability: Tensor, label: Tensor, condition: Callable[[int, int], bool]) -> int:
        probability = probability.squeeze()
        label = label.squeeze().reshape(-1)
        prediction = torch.argmax(probability, dim=0).reshape(-1)
        total: int = 0
        for prediction_pixel, label_pixel in zip(prediction, label):
            if condition(prediction_pixel, label_pixel):
                total += 1
        return total


class MergerNetwork(TemplateNetwork):
    def __init__(self, opt: BaseOptions, test_set: BaseDataset, train_set: BaseDataset):
        model: BaseModel = Merger(opt.channels_in, opt.height, opt.width, opt.number_of_classes)
        super().__init__(opt, test_set, train_set, model)


class UNetwork(TemplateNetwork):
    def __init__(self, opt: BaseOptions, test_set: BaseDataset, train_set: BaseDataset):
        model: BaseModel = UNet(opt.channels_in, opt.height, opt.width, opt.number_of_classes)
        super().__init__(opt, test_set, train_set, model)
