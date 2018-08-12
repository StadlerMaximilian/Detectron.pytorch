import argparse
import cv2
import os
import pprint
import sys
import pickle
import time

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import utils.logging as logging_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_dets', help='path of saved detections to load')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.',
        required=True)

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    return parser.parse_args()


def do_reval(dataset_name, output_dir, args):
    dataset = JsonDataset(dataset_name)
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = pickle.load(f)
    # Override config with the one saved in the detections file
    if args.cfg_file is not None:
        # bug: loads only already stored cfg
        # cfg.merge_cfg_from_cfg(core_config.load_cfg(dets['cfg']))
        # merge config from passed config file!!
        merge_cfg_from_file(args.cfg_file)
    else:
        cfg.merge_a_into_b(cfg.load_cfg(dets['cfg']), cfg)
    results = task_evaluation.evaluate_all(
        dataset,
        dets['all_boxes'],
        dets['all_segms'],
        dets['all_keyps'],
        output_dir,
        use_matlab=False)
    task_evaluation.log_copy_paste_friendly_results(results)


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = logging_utils.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    assert_and_infer_cfg()

    logger.info('Re-evaluating with config:')
    logger.info(pprint.pformat(cfg))

    # output_dir = os.path.abspath(args.output_dir[0])
    do_reval(args.dataset, args.output_dir, args)