#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import datetime

from core.config import cfg
from utils.logging import log_gan_stats
from utils.logging import SmoothedValue
from utils.timer import Timer

class ModeFlags(object):
    def __init__(self, mode, train):
        self.train_generator = True
        self.train_discriminator = False
        self.train_pre = False
        self.fake_mode = False
        self.real_mode = True

        self.set(mode, train)

    def set(self, mode, train):
        if mode == "fake":
            self.fake_mode = True
            self.real_mode = False
        elif mode == "real":
            self.real_mode = True
            self.fake_mode = False
        else:
            raise ValueError("mode ({}) has to be either 'real' or 'fake'!".format(mode))

        if train == "generator":
            self.train_generator = True
            self.train_discriminator = False
            self.train_pre = False
        elif train == "discriminator":
            self.train_generator = False
            self.train_discriminator = True
            self.train_pre = False
        elif train == "pre":
            self.train_generator = False
            self.train_discriminator = False
            self.train_pre = True
        else:
            raise ValueError("train ({}) has to be either 'generator' or 'discriminator'!".format(train))
        return self

    def __str__(self):
        return "\t fake \t real \n\t {}\t {}\n \t Gen \t Dis \n \t {} \t {}".format(self.fake_mode,
                                                                                    self.real_mode,
                                                                                    self.train_generator,
                                                                                    self.train_discriminator)


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, misc_args, log_period=20, max_iter=cfg.GAN.SOLVER.MAX_ITER, tensorboard_logger=None):
        # Output logging period in SGD iterations
        self.max_iter =max_iter
        self.misc_args = misc_args
        self.LOG_PERIOD = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['iter', 'eta']

        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 20

        def create_smoothed_value():
            return SmoothedValue(self.WIN_SZ)

        self.smoothed_losses = defaultdict(create_smoothed_value)
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        self.smoothed_metrics = defaultdict(create_smoothed_value)

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, out=None):
        """Update tracked iteration statistics."""
        if out is not None:  # first trained on either real/fake images (then set flag)
            total_loss = 0

            for k, loss in out['losses'].items():
                assert loss.shape[0] == cfg.NUM_GPUS
                loss = loss.mean(dim=0, keepdim=True)
                total_loss += loss
                loss_data = loss.data[0]
                out['losses'][k] = loss
                self.smoothed_losses[k].AddValue(loss_data)

            out['total_loss'] = total_loss  # Add the total loss for back propagation
            self.smoothed_total_loss.AddValue(total_loss.data[0])

            for k, metric in out['metrics'].items():
                metric = metric.mean(dim=0, keepdim=True)
                self.smoothed_metrics[k].AddValue(metric.data[0])

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0 or
                cur_iter == self.max_iter - 1):
            stats = self.GetStats(cur_iter, lr)
            log_gan_stats(stats, defaultdict(), defaultdict(), self.misc_args, self.max_iter)
            if self.tblogger:
                self.tb_log_stats(stats, cur_iter)

    def tb_log_stats(self, stats, cur_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self.tb_log_stats(v, cur_iter)
                else:
                    self.tblogger.add_scalar(k, v, cur_iter)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            self.max_iter - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            iter=cur_iter + 1,  # 1-indexed
            time=self.iter_timer.average_time,
            eta=eta,
            loss=self.smoothed_total_loss.GetMedianValue(),
            lr=lr,
        )
        stats['metrics'] = OrderedDict()
        for k in sorted(self.smoothed_metrics):
            stats['metrics'][k] = self.smoothed_metrics[k].GetMedianValue()

        head_losses = []
        adv_loss = []

        for k, v in self.smoothed_losses.items():
            toks = k.split('_')
            if len(toks) == 2 and toks[1] == 'adv':
                adv_loss.append((k, v.GetMedianValue()))
            elif len(toks) == 2:
                head_losses.append((k, v.GetMedianValue()))
            else:
                raise ValueError("Unexpected loss key: %s" % k)

        stats['head_losses'] = OrderedDict(head_losses)
        stats['adv_loss'] = OrderedDict(adv_loss)

        return stats
