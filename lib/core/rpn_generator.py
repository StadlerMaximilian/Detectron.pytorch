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
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Functions for RPN proposal generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import datetime
import logging
import numpy as np
import os
import yaml

import torch
from torch.autograd import Variable

from core.config import cfg
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
from utils.io import save_object
from utils.timer import Timer
from modeling.model_builder import Generalized_RCNN
import utils.net as net_utils
import nn as mynn
from utils.detectron_weight_helper import load_caffe2_detectron_weights
import utils.blob as blob_utils
import utils.env as envu
import utils.subprocess as subprocess_utils

logger = logging.getLogger(__name__)


def generate_rpn_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""

    output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        _boxes, _scores, _ids, rpn_file = multi_gpu_generate_rpn_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        # Processes entire dataset range by default
        _boxes, _scores, _ids, rpn_file = generate_rpn_on_range(
            args,
            dataset_name,
            proposal_file,
            output_dir,
            gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    return evaluate_proposal_file(dataset, rpn_file, output_dir)


def multi_gpu_generate_rpn_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    # Retrieve the test_net binary path
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    #TODO note that code can only be run from root_dir!!
    binary = os.path.join(binary_dir, 'tools/test_net' + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]

    # Run inference in parallel in subprocesses
    outputs = subprocess_utils.process_in_parallel(
        'rpn_proposals', num_images, binary, output_dir,
         args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    boxes, scores, ids = [], [], []
    for rpn_data in outputs:
        boxes += rpn_data['boxes']
        scores += rpn_data['scores']
        ids += rpn_data['ids']
    rpn_file = os.path.join(output_dir, 'rpn_proposals.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file
    )
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def generate_rpn_on_range(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert cfg.MODEL.RPN_ONLY or cfg.MODEL.FASTER_RCNN

    if not args.multi_gpu_testing:
        output_dir = os.path.join(output_dir, dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    roidb, start_ind, end_ind, total_num_images = get_roidb(
        dataset_name, ind_range
    )
    logger.info(
        'Output will be saved to: {:s}'.format(os.path.abspath(output_dir))
    )

    model = initialize_model_from_cfg(args, gpu_id=gpu_id)

    boxes, scores, ids = generate_proposals_on_roidb(
        model,
        roidb,
        start_ind=start_ind,
        end_ind=end_ind,
        total_num_images=total_num_images
    )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        rpn_name = 'rpn_proposals_range_{}_{}.pkl'.format(ind_range[0], ind_range[1])
    else:
        rpn_name = 'rpn_proposals.pkl'
    rpn_file = os.path.join(output_dir, rpn_name)
    save_object(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file
    )
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def generate_proposals_on_roidb(
    model, roidb, start_ind=None, end_ind=None, total_num_images=None
):
    """Generate RPN proposals on all images in an imdb."""
    _t = Timer()
    num_images = len(roidb)
    roidb_boxes = [[] for _ in range(num_images)]
    roidb_scores = [[] for _ in range(num_images)]
    roidb_ids = [[] for _ in range(num_images)]
    if start_ind is None:
        start_ind = 0
        end_ind = num_images
        total_num_images = num_images
    for i in range(num_images):
        roidb_ids[i] = roidb[i]['id']
        im = cv2.imread(roidb[i]['image'])
        _t.tic()
        roidb_boxes[i], roidb_scores[i] = im_proposals(model, im)
        _t.toc()
        if i % 10 == 0:
            ave_time = _t.average_time
            eta_seconds = ave_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                (
                    'rpn_generate: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, ave_time, eta
                )
            )

    return roidb_boxes, roidb_scores, roidb_ids


def im_proposals(model, im):
    """Generate RPN proposals on a single image."""

    inputs, im_scale = _get_blobs(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]

    blobs = model(**inputs)

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        raise NotImplementedError
    else:
        boxes = blobs['rpn_rois'].float()
        scores = blobs['rpn_roi_probs'].float()

    # Column 0 is the batch index in the (batch ind, x1, y1, x2, y2) encoding,
    # so we remove it since we just want to return boxes
    # Scale proposals back to the original input image scale
    if boxes.size()[1] == 4:
        boxes = boxes[:, :] / im_scale[0]
    else:
        boxes = boxes[:, 1:] / im_scale[0]
    return boxes, scores


def get_roidb(dataset_name, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, start, end, total_num_images


def evaluate_proposal_file(dataset, proposal_file, output_dir):
    """Evaluate box proposal average recall."""
    roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file)
    results = task_evaluation.evaluate_box_proposals(dataset, roidb)
    task_evaluation.log_box_proposal_results(results)
    recall_file = os.path.join(output_dir, 'rpn_proposal_{}_recall.pkl'.format(dataset.name))
    save_object(results, recall_file)
    return results


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_caffe2_detectron_weights(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def _get_blobs(im, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    return blobs, im_scale
