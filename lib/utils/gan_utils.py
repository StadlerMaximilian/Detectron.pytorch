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
import numpy as np

from core.config import cfg
from utils.logging import log_gan_stats
from utils.logging import SmoothedValue
from utils.timer import Timer
import utils.net as nu


class ModeFlags(object):
    def __init__(self, mode, train):
        self.train_generator = True
        self.train_discriminator = False
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
        elif train == "discriminator":
            self.train_generator = False
            self.train_discriminator = True
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

    def __init__(self, misc_args, log_period=20, tensorboard_logger=None):
        # Output logging period in SGD iterations
        self.misc_args = misc_args
        self.LOG_PERIOD = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['iter', 'eta']
        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 20

        def create_smoothed_value():
            return SmoothedValue(self.WIN_SZ)

        self.smoothed_losses_G = defaultdict(create_smoothed_value)
        self.smoothed_total_loss_G = SmoothedValue(self.WIN_SZ)
        self.smoothed_losses_D = defaultdict(create_smoothed_value)
        self.smoothed_metrics_D = defaultdict(create_smoothed_value)
        self.smoothed_metrics_G = defaultdict(create_smoothed_value)
        self.smoothed_total_loss_D = SmoothedValue(self.WIN_SZ)

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, out_D=None, out_G=None):
        """Update tracked iteration statistics."""

        if out_D is not None:  # first trained on either real/fake images (then set flag)
            total_loss = 0.0

            for k, loss in out_D['losses'].items():
                assert loss.shape[0] == cfg.NUM_GPUS
                loss = loss.mean(dim=0, keepdim=True)
                total_loss += loss
                loss_data = loss.data[0]
                out_D['losses'][k] = loss
                self.smoothed_losses_D[k].AddValue(loss_data)

            out_D['total_loss'] = total_loss  # Add the total loss for back propagation
            self.smoothed_total_loss_D.AddValue(total_loss.data[0])

            for k, metric in out_D['metrics'].items():
                metric = metric.mean(dim=0, keepdim=True)
                self.smoothed_metrics_D[k].AddValue(metric.data[0])

        elif out_G is not None:
            total_loss = 0.0

            for k, loss in out_G['losses'].items():
                assert loss.shape[0] == cfg.NUM_GPUS
                loss = loss.mean(dim=0, keepdim=True)
                total_loss += loss
                loss_data = loss.data[0]
                out_G['losses'][k] = loss
                self.smoothed_losses_G[k].AddValue(loss_data)

            out_G['total_loss'] = total_loss  # Add the total loss for back propagation
            self.smoothed_total_loss_G.AddValue(total_loss.data[0])

            for k, metric in out_G['metrics'].items():
                metric = metric.mean(dim=0, keepdim=True)
                self.smoothed_metrics_G[k].AddValue(metric.data[0])

    def LogIterStats(self, cur_iter, lr_D, lr_G):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0 or
                cur_iter == cfg.GAN.SOLVER.MAX_ITER - 1):
            stats = self.GetStats(cur_iter, lr_D, lr_G)
            log_gan_stats(stats, self.misc_args)
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

    def GetStats(self, cur_iter, lr_D, lr_G):
        eta_seconds = self.iter_timer.average_time * (
            cfg.SOLVER.MAX_ITER - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            iter=cur_iter + 1,  # 1-indexed
            time=self.iter_timer.average_time,
            eta=eta,
            loss_D=self.smoothed_total_loss_D.GetMedianValue(),
            loss_G=self.smoothed_total_loss_G.GetMedianValue(),
            lr_D=lr_D,
            lr_G=lr_G
        )
        stats['metrics'] = OrderedDict()
        for k in sorted(self.smoothed_metrics_D):
            stats['metrics'][k] = self.smoothed_metrics_D[k].GetMedianValue()

        head_losses_D = []
        head_losses_G = []
        adv_loss_D = []
        adv_loss_G = []

        for k, v in self.smoothed_losses_D.items():
            toks = k.split('_')
            if len(toks) == 2 and toks[1] == 'adv':
                adv_loss_D.append((k, v.GetMedianValue()))
            elif len(toks) == 2:
                head_losses_D.append((k, v.GetMedianValue()))
            else:
                raise ValueError("Unexpected loss key: %s" % k)

        for k, v in self.smoothed_losses_G.items():
            toks = k.split('_')
            if len(toks) == 2 and toks[1] == 'adv':
                adv_loss_G.append((k, v.GetMedianValue()))
            elif len(toks) == 2:
                head_losses_G.append((k, v.GetMedianValue()))
            else:
                raise ValueError("Unexpected loss key: %s" % k)

        stats['head_losses_D'] = OrderedDict(head_losses_D)
        stats['head_losses_G'] = OrderedDict(head_losses_G)
        stats['adv_loss_D'] = OrderedDict(adv_loss_D)
        stats['adv_loss_G'] = OrderedDict(adv_loss_G)

        return stats
