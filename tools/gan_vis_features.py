from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.gan_test import im_detect_all
from modeling.model_builder_gan import GAN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
from datasets.json_dataset import JsonDataset
from core.gan_test import _get_blobs
from utils.vis_feature_heat_map import show_heat_maps
from modeling.model_builder_gan import GAN
import utils.vis as vis_utils
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate GAN results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)

    parser.add_argument(
        '--pos',
        help='Flag for visualizing rectified activations.',
        action='store_true'
    )

    args = parser.parse_args()

    return args


def get_roidb_and_dataset(dataset_name, ind_range):
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

    return roidb, dataset, start, end, total_num_images


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = GAN()
    model.eval()
    model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def vis_features():
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.load_ckpt

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        args.dataset, args.range
    )

    assert_and_infer_cfg()
    gan = initialize_model_from_cfg(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, entry in enumerate(roidb):
        print("\t Image {} from {} ...".format(i + 1, len(roidb)))

        im = cv2.imread(entry['image'])

        inputs, im_scale = _get_blobs(im, None, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
            inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
        else:
            inputs['data'] = [torch.from_numpy(inputs['data'])]
            inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]

        output = gan(**inputs)

        output_np = defaultdict()

        for key in ['blob_conv_pooled', 'blob_fake', 'blob_conv_residual']:
            output_np[key] = output[key].data.cpu().numpy()
        output_np['rois'] = output['rpn_ret']['rois'].data.cpu().numpy()
        # scale images back to original image size
        output_np['rois'] = output_np['rois'][:, 1:5] / im_scale
        output_np['rois'] = output_np['rois'].astype(int)

        crop_img = [im[output_np['rois'][batch, 2]:output_np['rois'][batch, 4],
                    output_np['rois'][batch, 1]:output_np['rois'][batch, 3]] for batch in range(
            output_np['rois'].shape[0]
        )]
        show_heat_maps(output_np['blob_conv_pooled'], output_np['blob_fake'], output_np['blob_conv_residual'],
                       args.output_dir, "image_{}".format(i), blob_image=crop_img, ext="jpg", pos=args.pos)


if __name__ == '__main__':
    vis_features()

