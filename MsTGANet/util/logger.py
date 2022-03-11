from __future__ import annotations

import \
    sys
import time
from typing import Dict

import \
    numpy
from torch import Tensor
from visdom import Visdom

from MsTGANet.options.base_options import BaseOptions
from MsTGANet.util.utils import to_numpy, tensor_to_image


class Logger:
    def __init__(self, n_epochs: int, loss_plot_frequency: int):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.loss_plot_frequency = loss_plot_frequency
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, *, losses: Dict[str, Tensor] = dict(), images: Dict[str, Tensor] = dict()) -> None:
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        #sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] += losses[loss_name]

            # if (i+1) == len(losses.keys()):
            #     sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            # else:
            #     sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.loss_plot_frequency * (self.epoch - 1) + self.batch
        batches_left = self.loss_plot_frequency * (self.n_epochs - self.epoch) + self.loss_plot_frequency - self.batch
        #sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor_to_image(tensor), opts={'title': image_name})
            else:
                self.viz.image(tensor_to_image(tensor), win=self.image_windows[image_name], opts={'title': image_name})

        # End of epoch
        if (self.batch % self.loss_plot_frequency) == 0:
            # Plot losses
            self._plot_losses()
        else:
            self.batch += 1

    def _plot_losses(self) -> None:
        # plot losses
        for loss_name, loss in self.losses.items():
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=numpy.array([self.epoch]), Y=to_numpy(loss) / self.batch, opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
            else:
                self.viz.line(X=numpy.array([self.epoch]), Y=to_numpy(loss) / self.batch, win=self.loss_windows[loss_name], update='append')
            # reset losses for next epoch
            self.losses[loss_name] = 0.0

        self.epoch += 1
        self.batch = 1
        sys.stdout.write('\n')

