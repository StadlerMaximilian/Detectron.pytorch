"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
import re

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.gan_test_engine import run_inference
from modeling.model_builder_gan import GAN
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')

    parser.add_argument(
        '--load_dis', help='specific disriminator path, if no GAN model is loaded'
    )

    parser.add_argument(
        '--load_gen', help='specific generator path, if no GAN model is loaded'
    )

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi_gpu_testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


def test_net_routine(args):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    logger = utils.logging.setup_logging(__name__)

    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) or (bool(args.load_gen) and bool(args.load_dis))

    if args.output_dir is None:
        ckpt_path = args.load_ckpt
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    if args.load_dis is not None and args.load_gen is not None:
        dirs = args.load_gen.split('/')
        dirs = [x for x in dirs if x not in ['generator', 'ckpt']]
        dirs = [x for x in dirs if 'model_step' not in x]
        path_gan = os.path.join(dirs)
        _, file = os.path.split(args.load_gen)
        file = str(file.split('.')[0])
        step = int(re.findall(r'\d+', file)[0])
        path_gan = os.path.join(path_gan, 'ckpt')
        if not os.path.exists(path_gan):
            os.makedirs(path_gan)
        save_name = os.path.join(path_gan, 'model_step_{}.pth'.format(step))
        if os.path.exists(save_name):
            raise ValueError('CKPT already exists!!')
        gan = GAN(generator_weights=args.load_gen, discriminator_weights=args.load_dis)
        torch.save({'model': gan.state_dict()}, save_name)
        args.load_ckpt = save_name
        del gan

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)


if __name__ == '__main__':
    args = parse_args()
    test_net_routine(args)

