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

"""Utilities for logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime, timedelta
from collections import deque
from email.mime.text import MIMEText
import json
import logging
import numpy as np
import smtplib
import sys

from core.config import cfg

# Print lower precision floating point values than default FLOAT_REPR
# Note! Has no use for json encode with C speedups
json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')


def log_json_stats(stats, sort_keys=True):
    print('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))


def log_stats(stats, misc_args):
    """Log training statistics to terminal"""
    if hasattr(misc_args, 'epoch'):
        lines = "[%s][%s][Epoch %d][Iter %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename,
            misc_args.epoch, misc_args.step, misc_args.iters_per_epoch)
    else:
        lines = "[%s][%s][Step %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename, stats['iter'], cfg.SOLVER.MAX_ITER)

    lines += "\t\tloss: %.6f, lr: %.6f time: %.6f, eta: %s\n" % (
        stats['loss'], stats['lr'], stats['time'], stats['eta']
    )
    if stats['metrics']:
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['metrics'].items()) + "\n"
    if stats['head_losses']:
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['head_losses'].items()) + "\n"
    if cfg.RPN.RPN_ON:
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['rpn_losses'].items()) + "\n"
    if cfg.FPN.FPN_ON:
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['rpn_fpn_cls_losses'].items()) + "\n"
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['rpn_fpn_bbox_losses'].items()) + "\n"
    print(lines[:-1])  # remove last new line


def log_gan_stats(misc_args, max_iter, stats_gen=None, stats_dis=None, eta=None):
    """Log training statistics specifically for gans to terminal"""
    if hasattr(misc_args, 'epoch'):
        lines = "[%s][%s][Epoch %d][Iter %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename,
            misc_args.epoch, misc_args.step, misc_args.iters_per_epoch)
    else:
        lines = "[%s][%s][Step %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename, stats_dis['iter'], max_iter)

    if stats_gen is None and stats_dis is not None:
        lines += "\t\tlossDiscriminator: % .6f , lr_dis: %.6f time: %.6f, eta: %s\n" % (
                  stats_dis['loss'], stats_dis['lr'], stats_dis['time'], stats_dis['eta']
        )

        if stats_dis['metrics']:
            lines += "\t\tDiscriminator_metrics:  " + ", ".join("%s: %.6f" %
                                                      (k, v) for k, v in stats_dis['metrics'].items()) +  "\n"
        if stats_dis["head_losses"]:
            lines += "\t\tDiscriminator_head: " + ", ".join("%s: %.6f" %
                                                    (k, v) for k, v in stats_dis['head_losses'].items()) + "\n"
        if stats_dis['adv_loss']:
            lines += "\t\tDiscriminator_adv: " + ", ".join("%s: %.6f" %
                                                         (k, v) for k, v in stats_dis['adv_loss'].items()) + "\n"

    else:
        assert stats_dis is not None and stats_dis is not None and eta is not None

        lines += "\t\tloss_Generator: %.6f, loss_Discriminator: % .6f, " \
                 "lr_gen: %.6f, lr_dis: %.6f time_dis: %.6f, time_gen: %.6g, eta: %s\n" % (
                  stats_gen['loss'], stats_dis['loss'], stats_gen['lr'],
                  stats_dis['lr'], stats_dis['time'], stats_gen['time'], eta
                 )

        if stats_gen['metrics']:
            lines += "\t\tGenerator_metrics:" + ", ".join("%s: %.6f" % (k, v) for k, v in stats_gen['metrics'].items())

        if stats_dis['metrics']:
            lines += "\t\tDiscriminator_metrics " + ", ".join("%s: %.6f" %
                                                      (k, v) for k, v in stats_dis['metrics'].items()) +  "\n"
        if stats_gen['head_losses']:
            lines += "\t\tGenerator_head: " + ", ".join("%s: %.6f" %
                                                        (k, v) for k, v in stats_gen['head_losses'].items())
        if stats_dis["head_losses"]:
            lines += "\t\tDiscriminator_head: " + ", ".join("%s: %.6f" %
                                                     (k, v) for k, v in stats_dis['head_losses'].items()) + "\n"
        if stats_gen['adv_loss']:
            lines += "\t\tGenerator_adv: " + ", ".join("%s: %.6f" %
                                                       (k, v) for k, v in stats_gen['adv_loss'].items())
        if stats_dis['adv_loss']:
            lines += "\t\tDiscriminator_adv: " + ", ".join("%s: %.6f" %
                                                         (k, v) for k, v in stats_dis['adv_loss'].items()) + "\n"

    print(lines[:-1])  # remove last new line


def log_gan_stats_combined(cur_iter, lr_gen, lr_dis, training_stats_gen=None, training_stats_dis=None):
        if (cur_iter % training_stats_gen.LOG_PERIOD == 0 or
                cur_iter == training_stats_gen.max_iter - 1):

            eta_seconds_gen = training_stats_gen.iter_timer.average_time * (
                    training_stats_gen.max_iter - cur_iter
            )
            eta_seconds_dis = training_stats_dis.iter_timer.average_time * (
                    training_stats_dis.max_iter - cur_iter
            )

            eta = str(timedelta(seconds=int(eta_seconds_gen + eta_seconds_dis)))

            stats_gen = training_stats_gen.GetStats(cur_iter, lr_gen)
            stats_dis = training_stats_dis.GetStats(cur_iter, lr_dis)
            log_gan_stats(training_stats_gen.misc_args, training_stats_gen.max_iter,
                          stats_gen, stats_dis, eta)
            if training_stats_gen.tblogger:
                training_stats_gen.tb_log_stats(stats_gen, cur_iter)
            if training_stats_dis.tblogger:
                training_stats_dis.tb_log_stats(stats_dis, cur_iter)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count


def send_email(subject, body, to):
    s = smtplib.SMTP('localhost')
    mime = MIMEText(body)
    mime['Subject'] = subject
    mime['To'] = to
    s.sendmail('detectron', to, mime.as_string())


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger
