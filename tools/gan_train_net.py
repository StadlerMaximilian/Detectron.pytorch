""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
import time
from collections import defaultdict
from itertools import chain

import yaml
import torch
from torch.autograd import Variable
import cv2
from argparse import Namespace

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, \
    collate_minibatch_discriminator, collate_minibatch_generator, collate_minibatch_pre
from modeling.generator import Generator
from modeling.discriminator import Discriminator
from modeling.model_builder_gan import GAN
from utils.logging import setup_logging, log_gan_stats_combined
from utils.timer import Timer
from utils.gan_utils import TrainingStats, ModeFlags
from gan_test_net import test_net_routine

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--out', dest='output_dir', required=True,
        help='Root-Output_dir'
    )

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_pretrained', help='path to pretrained detectron model .pth',
    )

    parser.add_argument(
        '--load_ckpt', help='checkpoint path of GAN to load')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    parser.add_argument(
        '--bs_G', dest='batch_size_G',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)

    parser.add_argument(
        '--bs_D', dest='batch_size_D',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)

    parser.add_argument(
        '--bs_pre', dest='batch_size_pre',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)

    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--multi_gpu_testing',
        action='store_true',
        help='Flag for activating multi_gpu_testing'
    )

    parser.add_argument(
        '--init_dis_pretrained',
        action='store_true',
        help='Flag for initializing discriminator with pretrained weights. Else: pre-train on large objects'
    )

    parser.add_argument(
        '--online_cleanup',
        action='store_true',
        help='Flag for deleting objects and freeing GPU-cache, may increase run-time.'
    )

    return parser.parse_args()


def save_ckpt_gan(output_dir, args, step, train_size_gen, train_size_dis, model, optimizer_dis, optimizer_gen):
    if args.no_save:
        return

    batch_size_gen = args.batch_size_G
    batch_size_dis = args.batch_size_D

    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'train_size_gen': train_size_gen,
        'train_size_dis': train_size_dis,
        'batch_size_gen': batch_size_gen,
        'batch_size_dis': batch_size_dis,
        'model': model.state_dict(),
        'optimizer_gen': optimizer_gen.state_dict(),
        'optimizer_dis': optimizer_dis.state_dict()
    }, save_name)
    logger.info('save model: %s', save_name)
    return save_name


def save_ckpt(output_dir, args, step, train_size, model, optimizer, part="none"):
    """Save checkpoint"""
    if part == "G":
        batch_size = args.batch_size_G
    elif part == "D":
        batch_size = args.batch_size_D
    else:
        batch_size = args.batch_size

    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)
    return save_name


def save_model(output_dir, no_save, model):
    """Save final model"""
    if no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_gan_final.pth')
    if isinstance(model, mynn.DataParallel):
        model = model.module
    torch.save({
        'model': model.state_dict()
    }, save_name)
    logger.info('save model: %s', save_name)
    return save_name


def create_input_data(dataiterator, dataloader):
    try:
        input_data = next(dataiterator)
    except StopIteration:
        dataiterator = iter(dataloader)
        input_data = next(dataiterator)

    for key in input_data:
        if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
            input_data[key] = list(map(Variable, input_data[key]))

    return input_data, dataiterator


def main():
    """Main function"""
    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if cfg.RPN.RPN_ON:
        assert (args.load_pretrained is not None) | (args.load_ckpt is not None)
    else:
        assert (args.load_pretrained is not None) | (cfg.MODEL.LOAD_PRETRAINED_BACKBONE_WEIGHTS is not "") | \
               (args.load_ckpt is not None)

    if args.load_pretrained is not None and not os.path.exists(args.load_pretrained):
        raise ValueError("Specified pretrained detectron model does not exists")
    elif args.load_pretrained is not None:
        cfg.GAN.TRAIN.PRETRAINED_WEIGHTS = args.load_pretrained

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir

    # Adaptively adjust some configs
    original_num_gpus = cfg.NUM_GPUS
    cfg.NUM_GPUS = torch.cuda.device_count()

    # Adaptively adjust some configs for the PRE-TRAINING
    original_batch_size_pre = cfg.NUM_GPUS * cfg.GAN.TRAIN.IMS_PER_BATCH_PRE
    original_ims_per_batch_pre = cfg.GAN.TRAIN.IMS_PER_BATCH_PRE
    if args.batch_size_pre is None:
        args.batch_size_pre = original_batch_size_pre
    assert (args.batch_size_pre % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size_pre, cfg.NUM_GPUS)
    cfg.GAN.TRAIN.IMS_PER_BATCH_PRE = args.batch_size_pre // cfg.NUM_GPUS
    effective_batch_size_pre = args.iter_size * args.batch_size_pre
    print('effective_batch_size_pre = batch_size * iter_size = %d * %d' % (args.batch_size_pre, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size_pre, effective_batch_size_pre))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch_pre, cfg.GAN.TRAIN.IMS_PER_BATCH_PRE))

    # Adaptively adjust some configs for discriminator #
    original_batch_size_D = cfg.NUM_GPUS * cfg.GAN.TRAIN.IMS_PER_BATCH_D
    original_ims_per_batch_D = cfg.GAN.TRAIN.IMS_PER_BATCH_D
    if args.batch_size_D is None:
        args.batch_size_D = original_batch_size_D
    assert (args.batch_size_D % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size_D, cfg.NUM_GPUS)
    cfg.GAN.TRAIN.IMS_PER_BATCH_D = args.batch_size_D // cfg.NUM_GPUS
    effective_batch_size_D = args.iter_size * args.batch_size_D
    print('effective_batch_size_D = batch_size * iter_size = %d * %d' % (args.batch_size_D, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size_D, effective_batch_size_D))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch_D, cfg.GAN.TRAIN.IMS_PER_BATCH_D))

    # Adaptively adjust some configs for generator #
    original_batch_size_G = cfg.NUM_GPUS * cfg.GAN.TRAIN.IMS_PER_BATCH_G
    original_ims_per_batch_G = cfg.GAN.TRAIN.IMS_PER_BATCH_G
    if args.batch_size_G is None:
        args.batch_size_G = original_batch_size_G
    assert (args.batch_size_G % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size_G, cfg.NUM_GPUS)
    cfg.GAN.TRAIN.IMS_PER_BATCH_G = args.batch_size_G // cfg.NUM_GPUS
    effective_batch_size_G = args.iter_size * args.batch_size_G
    print('effective_batch_size_G = batch_size * iter_size = %d * %d' % (args.batch_size_G, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size_G, effective_batch_size_G))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch_G, cfg.GAN.TRAIN.IMS_PER_BATCH_G))

    # Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr_D = cfg.GAN.SOLVER.BASE_LR_D
    old_base_lr_G = cfg.GAN.SOLVER.BASE_LR_G
    old_base_lr_pre = cfg.GAN.SOLVER.BASE_LR_PRE
    cfg.GAN.SOLVER.BASE_LR_D *= args.batch_size_D / original_batch_size_D
    cfg.GAN.SOLVER.BASE_LR_PRE *= args.batch_size_pre / original_batch_size_pre
    cfg.GAN.SOLVER.BASE_LR_G *= args.batch_size_G / original_batch_size_G
    print('Adjust BASE_LR_PRE linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr_pre, cfg.GAN.SOLVER.BASE_LR_PRE))
    print('Adjust BASE_LR_D linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr_D, cfg.GAN.SOLVER.BASE_LR_D))
    print('Adjust BASE_LR_G linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr_G, cfg.GAN.SOLVER.BASE_LR_G))

    # Adjust solver steps
    step_scale_pre = original_batch_size_pre / effective_batch_size_pre
    step_scale_D = original_batch_size_D / effective_batch_size_D
    step_scale_G = original_batch_size_G / effective_batch_size_G
    if not cfg.GAN.SOLVER.STEPS_D:
        cfg.GAN.SOLVER.STEPS_D = cfg.GAN.SOLVER.STEPS
    if not cfg.GAN.SOLVER.STEPS_G:
        cfg.GAN.SOLVER.STEPS_G = cfg.GAN.SOLVER.STEPS
    old_solver_steps_D = cfg.GAN.SOLVER.STEPS_D
    old_solver_steps_G = cfg.GAN.SOLVER.STEPS_G
    old_solver_steps_pre = cfg.GAN.SOLVER.STEPS_PRE
    old_max_iter = cfg.GAN.SOLVER.MAX_ITER
    old_max_iter_pre = cfg.GAN.SOLVER.PRE_ITER
    cfg.GAN.SOLVER.STEPS_PRE = list(map(lambda x: int(x * step_scale_pre + 0.5), cfg.GAN.SOLVER.STEPS_PRE))
    cfg.GAN.SOLVER.STEPS_D = list(map(lambda x: int(x * step_scale_D + 0.5), cfg.GAN.SOLVER.STEPS_D))
    cfg.GAN.SOLVER.STEPS_G = list(map(lambda x: int(x * step_scale_G + 0.5), cfg.GAN.SOLVER.STEPS_G))
    cfg.GAN.SOLVER.MAX_ITER_D = int(cfg.GAN.SOLVER.MAX_ITER * step_scale_D + 0.5)
    cfg.GAN.SOLVER.MAX_ITER_G = int(cfg.GAN.SOLVER.MAX_ITER * step_scale_G + 0.5)
    cfg.GAN.SOLVER.PRE_ITER = int(cfg.GAN.SOLVER.PRE_ITER * step_scale_pre + 0.5)
    print('PRE: Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps_pre, cfg.GAN.SOLVER.STEPS_PRE,
                                                  old_max_iter_pre, cfg.GAN.SOLVER.PRE_ITER))
    print('DIS: Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps_D, cfg.GAN.SOLVER.STEPS_D,
                                                  old_max_iter, cfg.GAN.SOLVER.MAX_ITER_D))
    print('GEN: Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps_G, cfg.GAN.SOLVER.STEPS_G,
                                                  old_max_iter, cfg.GAN.SOLVER.MAX_ITER_G))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    assert_and_infer_cfg(make_immutable=False)

    timers = defaultdict(Timer)

    # prepare flags
    # for FAST R-CNN: rois are not sampled on the run. The flags therefore have to be passed to the actual dataloader
    fake_dis_flag = [ModeFlags("fake", "discriminator") for _ in range(cfg.NUM_GPUS)]
    real_dis_flag = [ModeFlags("real", "discriminator") for _ in range(cfg.NUM_GPUS)]
    fake_gen_flag = [ModeFlags("fake", "generator") for _ in range(cfg.NUM_GPUS)]
    pre_flag = [ModeFlags("real", "pre") for _ in range(cfg.NUM_GPUS)]

    # Datasets #
    timers['roidb_real'].tic()
    roidb_real, ratio_list_real, ratio_index_real = combined_roidb_for_training(
        cfg.GAN.TRAIN.DATASETS_REAL, cfg.GAN.TRAIN.PROPOSAL_FILES_REAL)
    timers['roidb_real'].toc()
    roidb_size_real = len(roidb_real)
    logger.info('{:d} roidb entries'.format(roidb_size_real))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb_real'].average_time)

    # Effective training sample size for one epoch
    train_size_D = roidb_size_real // args.batch_size_D * args.batch_size_D

    batchSampler_pre = BatchSampler(
        sampler=MinibatchSampler(ratio_list_real, ratio_index_real, cfg.GAN.TRAIN.IMS_PER_BATCH_PRE),
        batch_size=args.batch_size_pre,
        drop_last=True
    )

    dataset_pre = RoiDataLoader(
        roidb_real,
        cfg.MODEL.NUM_CLASSES,
        training=True,
        flags=pre_flag[0])

    dataloader_pre = torch.utils.data.DataLoader(
        dataset_pre,
        batch_sampler=batchSampler_pre,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch_pre,
        pin_memory=False)

    dataiterator_pre = iter(dataloader_pre)

    batchSampler_real_discriminator= BatchSampler(
        sampler=MinibatchSampler(ratio_list_real, ratio_index_real, cfg.GAN.TRAIN.IMS_PER_BATCH_D),
        batch_size=args.batch_size_D,
        drop_last=True
    )

    dataset_real_discriminator = RoiDataLoader(
        roidb_real,
        cfg.MODEL.NUM_CLASSES,
        training=True,
        flags=real_dis_flag[0])

    dataloader_real_discriminator = torch.utils.data.DataLoader(
        dataset_real_discriminator,
        batch_sampler=batchSampler_real_discriminator,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch_discriminator,
        pin_memory=False)

    dataiterator_real_discriminator = iter(dataloader_real_discriminator)

    timers['roidb_fake'].tic()
    roidb_fake, ratio_list_fake, ratio_index_fake = combined_roidb_for_training(
        cfg.GAN.TRAIN.DATASETS_FAKE, cfg.GAN.TRAIN.PROPOSAL_FILES_FAKE)
    timers['roidb_fake'].toc()
    roidb_size_fake = len(roidb_fake)
    logger.info('{:d} roidb entries'.format(roidb_size_fake))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb_fake'].average_time)

    # Effective training sample size for one epoch
    train_size_G = roidb_size_fake // args.batch_size_G * args.batch_size_G

    batchSampler_fake_discriminator = BatchSampler(
        sampler=MinibatchSampler(ratio_list_fake, ratio_index_fake, cfg.GAN.TRAIN.IMS_PER_BATCH_D),
        batch_size=args.batch_size_D,
        drop_last=True
    )

    dataset_fake_discriminator = RoiDataLoader(
        roidb_fake,
        cfg.MODEL.NUM_CLASSES,
        training=True,
        flags=fake_dis_flag[0]
    )

    dataloader_fake_discriminator = torch.utils.data.DataLoader(
        dataset_fake_discriminator,
        batch_sampler=batchSampler_fake_discriminator,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch_discriminator,
        pin_memory=False)

    dataiterator_fake_discriminator = iter(dataloader_fake_discriminator)

    batchSampler_fake_generator = BatchSampler(
        sampler=MinibatchSampler(ratio_list_fake, ratio_index_fake, cfg.GAN.TRAIN.IMS_PER_BATCH_G),
        batch_size=args.batch_size_G,
        drop_last=True
    )

    dataset_fake_generator = RoiDataLoader(
        roidb_fake,
        cfg.MODEL.NUM_CLASSES,
        training=True,
        flags=fake_gen_flag[0]
    )

    dataloader_fake_generator = torch.utils.data.DataLoader(
        dataset_fake_generator,
        batch_sampler=batchSampler_fake_generator,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch_generator,
        pin_memory=False)

    dataiterator_fake_generator = iter(dataloader_fake_generator)
    train_size = max(train_size_D // 2, train_size_G)

    # Model
    # only load pre-trained discriminator explicitly specified
    if cfg.MODEL.FASTER_RCNN:
        if args.init_dis_pretrained:
            gan = GAN(generator_weights=cfg.GAN.TRAIN.PRETRAINED_WEIGHTS,
                      discriminator_weights=cfg.GAN.TRAIN.PRETRAINED_WEIGHTS)
        else:
            gan = GAN(generator_weights=cfg.GAN.TRAIN.PRETRAINED_WEIGHTS)
    else: # if Fast R-CNN, start with new model, but use pre-trained weights from config (on ImageNet)
        gan = GAN()

    if cfg.CUDA:
        gan.cuda()

    params_list_D = {
        'bias_params': [],
        'bias_param_names': [],
        'nonbias_params': [],
        'nonbias_param_names': [],
        'nograd_param_names': []
    }

    # train discriminator only on adversarial branch
    for key, value in gan.discriminator.adversarial.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params_list_D['bias_params'].append(value)
                params_list_D['bias_param_names'].append(key)
            else:
                params_list_D['nonbias_params'].append(value)
                params_list_D['nonbias_param_names'].append(key)
        else:
            params_list_D['nograd_param_names'].append(key)

    params_D = [
        {'params': params_list_D['nonbias_params'],
         'lr': 0,
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_D},
        {'params': params_list_D['bias_params'],
         'lr': 0 * (cfg.GAN.SOLVER.BIAS_DOUBLE_LR_D + 1),
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_D if cfg.GAN.SOLVER.BIAS_WEIGHT_DECAY_D else 0}
    ]
    param_names_D = [params_list_D['nonbias_param_names'], params_list_D['bias_param_names']]

    logger.info("Parameters discriminator is trained on")
    logger.info(param_names_D)

    params_list_pre = {
        'bias_params': [],
        'bias_param_names': [],
        'nonbias_params': [],
        'nonbias_param_names': [],
        'nograd_param_names': []
    }

    # pre-train either on perceptual branch and/or Generator_block for Faster R-CNN
    # or pre-train on perceptual branch and conv_body for fast r-cnn

    if cfg.MODEL.FASTER_RCNN:
        if cfg.GAN.TRAIN.PRE_TRAIN_GENERATOR:
            pre_named_params = chain(gan.discriminator.Box_Head.named_parameters(),
                                     gan.discriminator.Box_Outs.named_parameters(),
                                     gan.generator.Generator_Block.named_parameters())
        else:
            pre_named_params = chain(gan.discriminator.Box_Head.named_parameters(),
                                     gan.discriminator.Box_Outs.named_parameters())
    else:
        pre_named_params = chain(gan.discriminator.Box_Head.named_parameters(),
                                 gan.discriminator.Box_Outs.named_parameters(),
                                 gan.generator.Conv_Body.named_parameters())

    for key, value in pre_named_params:
        if value.requires_grad:
            if 'bias' in key:
                params_list_pre['bias_params'].append(value)
                params_list_pre['bias_param_names'].append(key)
            else:
                params_list_pre['nonbias_params'].append(value)
                params_list_pre['nonbias_param_names'].append(key)
        else:
            params_list_pre['nograd_param_names'].append(key)

    params_pre = [
        {'params': params_list_pre['nonbias_params'],
         'lr': 0,
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_PRE},
        {'params': params_list_pre['bias_params'],
         'lr': 0 * (cfg.GAN.SOLVER.BIAS_DOUBLE_LR_PRE + 1),
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_PRE if cfg.GAN.SOLVER.BIAS_WEIGHT_DECAY_PRE else 0}
    ]
    param_names_pre = [params_list_pre['nonbias_param_names'], params_list_pre['bias_param_names']]

    logger.info("Parameters during pre-training")
    logger.info(param_names_pre)

    params_list_G = {
        'bias_params': [],
        'bias_param_names': [],
        'nonbias_params': [],
        'nonbias_param_names': [],
        'nograd_param_names': []
    }

    # train generator only on generator network (generator block)
    for key, value in gan.generator.Generator_Block.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params_list_G['bias_params'].append(value)
                params_list_G['bias_param_names'].append(key)
            else:
                params_list_G['nonbias_params'].append(value)
                params_list_G['nonbias_param_names'].append(key)
        else:
            params_list_G['nograd_param_names'].append(key)

    params_G = [
        {'params': params_list_G['nonbias_params'],
         'lr': 0,
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_G},
        {'params': params_list_G['bias_params'],
         'lr': 0 * (cfg.GAN.SOLVER.BIAS_DOUBLE_LR_G + 1),
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_G if cfg.GAN.SOLVER.BIAS_WEIGHT_DECAY_G else 0}
    ]

    param_names_G = [params_list_G['nonbias_param_names'], params_list_G['bias_param_names']]
    logger.info("Parameters generator is trained on")
    logger.info(param_names_G)

    # Optimizers
    if cfg.GAN.SOLVER.TYPE_G == "SGD":
        optimizer_G = torch.optim.SGD(params_G, momentum=cfg.GAN.SOLVER.MOMENTUM_G)
    elif cfg.GAN.SOLVER.TYPE_G == "Adam":
        optimizer_G = torch.optim.Adam(params_G)
    else:
        raise ValueError("INVALID Optimizer_G specified. Must be SGD or Adam!")

    if cfg.GAN.SOLVER.TYPE_D == "SGD":
        optimizer_D = torch.optim.SGD(params_D, momentum=cfg.GAN.SOLVER.MOMENTUM_D)
    elif cfg.GAN.SOLVER.TYPE_D == "Adam":
        optimizer_D = torch.optim.Adam(params_D)
    else:
        raise ValueError("INVALID Optimizer_D specified. Must be SGD or Adam!")

    if cfg.GAN.SOLVER.TYPE_PRE == "SGD":
        optimizer_pre = torch.optim.SGD(params_pre, momentum=cfg.GAN.SOLVER.MOMENTUM_PRE)
    elif cfg.GAN.SOLVER.TYPE_PRE == "Adam":
        optimizer_pre = torch.optim.Adam(params_pre)
    else:
        raise ValueError("INVALID Optimizer_pre specified. Must be SGD or Adam!")

    # Load checkpoint
    # loading checkpoint is only possible for combined gan training
    if args.load_ckpt:
        args.init_dis_pretrained = True
        load_name = args.load_ckp
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(gan, checkpoint['model'])

        if args.resume:
            # as for every k steps of discriminator, G is updated once
            optimizer_G.load_state_dict(checkpoint['optimizer_gen'])
            optimizer_D.load_state_dict(checkpoint['optimizer_dis'])
        del checkpoint
        torch.cuda.empty_cache()

    lr_D = optimizer_D.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.
    lr_G = optimizer_G.param_groups[0]['lr']
    lr_pre = optimizer_pre.param_groups[0]['lr']

    gan = mynn.DataParallel(gan, cpu_keywords=['im_info', 'roidb'],
                            minibatch=True)

    # Training Setups
    args.run_name = misc_utils.get_run_name() + '_step'
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    output_dir_pre = os.path.join(output_dir, 'pre')
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir_pre):
            os.makedirs(output_dir_pre)
        logging.info("Using output_dir: {}".format(output_dir))

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger_dis_real= SummaryWriter(os.path.join(output_dir, 'log', 'real'),
                                             filename_suffix="_discriminator_real")
            tblogger_dis_fake = SummaryWriter(os.path.join(output_dir, 'log', 'fake'),
                                              filename_suffix="_discriminator_fake")
            tblogger_gen = SummaryWriter(os.path.join(output_dir, 'log', 'gen'),
                                         filename_suffix="_generator")
            tblogger_pre = SummaryWriter(os.path.join(output_dir_pre, 'log', 'pre'),
                                         filename_suffix="_pre")

    ### Training Loop ###
    gan.train()

    CHECKPOINT_PERIOD = int(cfg.GAN.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    # Set index for decay steps
    decay_steps_ind_D = None
    decay_steps_ind_G = None
    decay_steps_ind_pre = None
    for i in range(1, len(cfg.GAN.SOLVER.STEPS_D)):
        if cfg.GAN.SOLVER.STEPS_D[i] >= args.start_step:
            decay_steps_ind_D = i
            break
    if decay_steps_ind_D is None:
        decay_steps_ind_D = len(cfg.GAN.SOLVER.STEPS_D)

    for i in range(1, len(cfg.GAN.SOLVER.STEPS_G)):
        if cfg.GAN.SOLVER.STEPS_G[i] >= args.start_step:
            decay_steps_ind_G = i
            break
    if decay_steps_ind_G is None:
        decay_steps_ind_G = len(cfg.GAN.SOLVER.STEPS_G)

    for i in range(1, len(cfg.GAN.SOLVER.STEPS_PRE)):
        if cfg.GAN.SOLVER.STEPS_PRE[i] >= args.start_step:
            decay_steps_ind_pre = i
            break
    if decay_steps_ind_pre is None:
        decay_steps_ind_pre = len(cfg.GAN.SOLVER.STEPS_PRE)

    training_stats_pre = TrainingStats(
        args,
        args.disp_interval,
        cfg.GAN.SOLVER.PRE_ITER,
        tblogger_pre if args.use_tfboard and not args.no_save else None)

    # use maximum max_iter for training
    max_iter = max(cfg.GAN.SOLVER.MAX_ITER_D, cfg.GAN.SOLVER.MAX_ITER_G)

    try:
        logger.info('Training starts !')
        step = args.start_step

        # prepare adv_targets for training
        Tensor = torch.cuda.FloatTensor
        batch_size = cfg.GAN.TRAIN.IMS_PER_BATCH_D * cfg.GAN.TRAIN.BATCH_SIZE_PER_IM_D
        batch_size_gen = cfg.GAN.TRAIN.IMS_PER_BATCH_G * cfg.GAN.TRAIN.BATCH_SIZE_PER_IM_G
        batch_size_pre = cfg.GAN.TRAIN.IMS_PER_BATCH_PRE * cfg.GAN.TRAIN.BATCH_SIZE_PER_IM_PRE

        adv_target_real = [Variable(Tensor(batch_size, 1).fill_(cfg.GAN.MODEL.LABEL_SMOOTHING),
                                    requires_grad=False) for _ in range(cfg.NUM_GPUS)]
        adv_target_gen = [Variable(Tensor(batch_size_gen, 1).fill_(cfg.GAN.MODEL.LABEL_SMOOTHING),
                                   requires_grad=False) for _ in range(cfg.NUM_GPUS)]
        adv_target_pre = [Variable(Tensor(batch_size_pre, 1).fill_(cfg.GAN.MODEL.LABEL_SMOOTHING),
                                   requires_grad=False) for _ in range(cfg.NUM_GPUS)]
        adv_target_fake = [Variable(Tensor(batch_size, 1).fill_(0.0),
                                    requires_grad=False) for _ in range(cfg.NUM_GPUS)]

        # pre-training of perceptual branch
        if not args.init_dis_pretrained:
            logger.info('Pre-Training: training perceptual-branch on large objects')

            for step in range(0, cfg.GAN.SOLVER.PRE_ITER):
                # Warm up
                # for simplicity: equal for generator and discriminator
                if step < cfg.GAN.SOLVER.PRE_WARM_UP_ITERS:
                    method = cfg.GAN.SOLVER.WARM_UP_METHOD
                    if method == 'constant':
                        warmup_factor = cfg.GAN.SOLVER.WARM_UP_FACTOR
                    elif method == 'linear':
                        alpha = step / cfg.GAN.SOLVER.PRE_WARM_UP_ITERS
                        warmup_factor = cfg.GAN.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                    else:
                        raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                    lr_new_pre = cfg.GAN.SOLVER.BASE_LR_PRE * warmup_factor
                    net_utils.update_learning_rate(optimizer_pre, lr_pre, lr_new_pre, type='pre')
                    lr_pre = optimizer_pre.param_groups[0]['lr']
                    assert lr_pre == lr_new_pre
                elif step == cfg.GAN.SOLVER.PRE_WARM_UP_ITERS :
                    net_utils.update_learning_rate(optimizer_pre, lr_pre, cfg.GAN.SOLVER.BASE_LR_PRE, type='pre')
                    lr_pre = optimizer_pre.param_groups[0]['lr']
                    assert lr_pre == cfg.GAN.SOLVER.BASE_LR_PRE

                # Learning rate decay
                if decay_steps_ind_pre < len(cfg.GAN.SOLVER.STEPS_PRE) and \
                        step == cfg.GAN.SOLVER.STEPS_PRE[decay_steps_ind_pre]:
                    logger.info('Decay the learning (pre-training) on step %d', step)
                    lr_new_pre = lr_pre * cfg.GAN.SOLVER.GAMMA_PRE
                    net_utils.update_learning_rate(optimizer_pre, lr_pre, lr_new_pre, type='pre')
                    lr_pre = optimizer_pre.param_groups[0]['lr']
                    assert lr_pre == lr_new_pre
                    decay_steps_ind_pre += 1

                optimizer_pre.zero_grad()
                training_stats_pre.IterTic()

                input_data_pre, dataiterator_pre = create_input_data(
                    dataiterator_pre, dataloader_pre
                )

                input_data_pre.update({"flags": pre_flag,
                                       "adv_target": adv_target_pre}
                                      )
                outputs_pre = gan(**input_data_pre)
                # only train perceptual branch
                # remove adv loss
                training_stats_pre.UpdateIterStats(outputs_pre)
                # train only on the Perceptual Branch
                loss_pre = outputs_pre['losses']['loss_cls'] + outputs_pre['losses']['loss_bbox']
                loss_pre.backward()
                optimizer_pre.step()

                training_stats_pre.IterToc()
                training_stats_pre.LogIterStatsReal(step, lr=lr_pre)

            # clean up
            if args.use_tfboard and not args.no_save:
                tblogger_pre.close()

            del dataiterator_pre
            del dataloader_pre
            del batchSampler_pre
            del dataset_pre
            del training_stats_pre
            del input_data_pre
            del loss_pre
            del training_stats_pre
            torch.cuda.empty_cache()

            # save model after pre-training
            save_ckpt_gan(output_dir_pre, args, step, train_size_gen=train_size_G, train_size_dis=train_size_D,
                          model=gan, optimizer_dis=optimizer_pre, optimizer_gen=optimizer_G)

        # combined training
        training_stats_dis_real = TrainingStats(
            args,
            args.disp_interval,
            max_iter,
            tblogger_dis_real if args.use_tfboard and not args.no_save else None)

        training_stats_dis_fake = TrainingStats(
            args,
            args.disp_interval,
            max_iter,
            tblogger_dis_fake if args.use_tfboard and not args.no_save else None)

        training_stats_gen = TrainingStats(
            args,
            args.disp_interval,
            max_iter,
            tblogger_gen if args.use_tfboard and not args.no_save else None)

        logger.info('Combined GAN-training starts now!')
        for step in range(args.start_step, max_iter):

            # Warm up
            # for simplicity: equal for generator and discriminator
            if step < cfg.GAN.SOLVER.WARM_UP_ITERS:
                method = cfg.GAN.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.GAN.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.GAN.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.GAN.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new_D = cfg.GAN.SOLVER.BASE_LR_D * warmup_factor
                lr_new_G = cfg.GAN.SOLVER.BASE_LR_G * warmup_factor
                net_utils.update_learning_rate(optimizer_D, lr_D, lr_new_D)
                net_utils.update_learning_rate(optimizer_G, lr_G, lr_new_G)
                lr_D = optimizer_D.param_groups[0]['lr']
                lr_G = optimizer_G.param_groups[0]['lr']
                assert lr_D == lr_new_D
                assert lr_G == lr_new_G
            elif step == cfg.GAN.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(optimizer_D, lr_D, cfg.GAN.SOLVER.BASE_LR_D)
                net_utils.update_learning_rate(optimizer_G, lr_G, cfg.GAN.SOLVER.BASE_LR_G)
                lr_D = optimizer_D.param_groups[0]['lr']
                lr_G = optimizer_G.param_groups[0]['lr']
                assert lr_D == cfg.GAN.SOLVER.BASE_LR_D
                assert lr_G == cfg.GAN.SOLVER.BASE_LR_G

            # Learning rate decay
            if decay_steps_ind_D < len(cfg.GAN.SOLVER.STEPS_D) and \
                    step == cfg.GAN.SOLVER.STEPS_D[decay_steps_ind_D]:
                logger.info('Decay the learning (discriminator) on step %d', step)
                lr_new_D = lr_D * cfg.GAN.SOLVER.GAMMA_D
                net_utils.update_learning_rate(optimizer_D, lr_D, lr_new_D)
                lr_D = optimizer_D.param_groups[0]['lr']
                assert lr_D == lr_new_D
                decay_steps_ind_D += 1

            if decay_steps_ind_G < len(cfg.GAN.SOLVER.STEPS_G) and \
                    step == cfg.GAN.SOLVER.STEPS_G[decay_steps_ind_G]:
                logger.info('Decay the learning (generator) on step %d', step)
                lr_new_G = lr_G * cfg.GAN.SOLVER.GAMMA_G
                net_utils.update_learning_rate(optimizer_G, lr_G, lr_new_G)
                lr_G = optimizer_G.param_groups[0]['lr']
                assert lr_G == lr_new_G
                decay_steps_ind_G += 1

            # train discriminator only on adversarial branch
            for _ in range(cfg.GAN.TRAIN.k):

                optimizer_D.zero_grad()
                training_stats_dis_fake.IterTic()
                training_stats_dis_real.IterTic()

                # train on fake data

                input_data, dataiterator_fake_discriminator = create_input_data(
                    dataiterator_fake_discriminator, dataloader_fake_discriminator
                )

                input_data.update({"flags": fake_dis_flag,
                                   "adv_target": adv_target_fake}
                                  )
                outputs = gan(**input_data)
                training_stats_dis_fake.UpdateIterStats(out=outputs)
                loss_fake = outputs['losses']['loss_adv']  # adversarial loss for discriminator
                training_stats_dis_fake.tb_log_stats(training_stats_dis_fake.GetStats(step, lr_D), step)

                # train on real data
                input_data, dataiterator_real_discriminator = create_input_data(
                    dataiterator_real_discriminator, dataloader_real_discriminator
                )

                input_data.update({"flags": real_dis_flag,
                                   "adv_target": adv_target_real}
                                  )
                outputs = gan(**input_data)
                training_stats_dis_real.UpdateIterStats(out=outputs)
                loss_real = outputs['losses']['loss_adv']  # adversarial loss for discriminator
                training_stats_dis_real.tb_log_stats(training_stats_dis_real.GetStats(step, lr_D), step)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizer_D.step()

                training_stats_dis_fake.IterToc()
                training_stats_dis_real.IterToc()

            # train generator on total loss of discriminator (all losses weighted simply with 1)
            training_stats_gen.IterTic()

            optimizer_G.zero_grad()
            input_data, dataiterator_fake_generator = create_input_data(
                dataiterator_fake_generator, dataloader_fake_generator
            )

            input_data.update({"flags": fake_gen_flag,
                               "adv_target": adv_target_gen}
                              )
            outputs = gan(**input_data)
            training_stats_gen.UpdateIterStats(out=outputs)
            training_stats_gen.tb_log_stats(training_stats_gen.GetStats(step, lr_G), step)

            # train generator on Faster R-CNN loss and adversarial loss
            loss_G = outputs['losses']['loss_cls'] + outputs['losses']['loss_bbox']
            loss_G = loss_G + outputs['losses']['loss_adv']
            loss_G.backward()
            optimizer_G.step()
            training_stats_gen.IterToc()

            log_gan_stats_combined(step, lr_gen=lr_G, lr_dis=lr_D,
                                   training_stats_dis_real=training_stats_dis_real,
                                   training_stats_gen=training_stats_gen,
                                   training_stats_dis_fake=training_stats_dis_fake)

            if (step+1) % CHECKPOINT_PERIOD == 0:
                save_ckpt_gan(output_dir, args, step, train_size_gen=train_size_G, train_size_dis=train_size_D,
                              model=gan, optimizer_dis=optimizer_D, optimizer_gen=optimizer_G)

        # ---- Training ends ----
        # Save last checkpoint
        final_model = save_ckpt_gan(output_dir, args, step, train_size_gen=train_size_G, train_size_dis=train_size_D,
                                    model=gan, optimizer_dis=optimizer_D, optimizer_gen=optimizer_G)
        # cleanup
        del gan
        del loss_G
        del loss_D
        del loss_real
        del loss_fake
        del input_data
        del dataiterator_real_discriminator
        del dataiterator_fake_discriminator
        del dataiterator_fake_generator
        del dataloader_fake_discriminator
        del dataloader_fake_generator
        del dataloader_real_discriminator
        del batchSampler_fake_discriminator
        del batchSampler_fake_generator
        del batchSampler_real_discriminator
        del dataset_fake_discriminator
        del dataset_real_discriminator
        del dataset_fake_generator
        del optimizer_G
        del optimizer_D
        torch.cuda.empty_cache()

        logger.info("Closing dataloader and tfboard if used")
        if args.use_tfboard and not args.no_save:
            tblogger_dis_fake.close()
            tblogger_dis_real.close()
            tblogger_gen.close()

        del training_stats_dis_fake
        del training_stats_dis_real
        del training_stats_gen

    except (RuntimeError, KeyboardInterrupt):

        del dataiterator_real_discriminator
        del dataiterator_fake_discriminator
        del dataiterator_fake_generator

        logger.info('Save ckpt on exception ...')

        save_ckpt_gan(output_dir, args, step, train_size_gen=train_size_G, train_size_dis=train_size_D,
                      model=gan, optimizer_dis=optimizer_D, optimizer_gen=optimizer_G)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)
        logger.info("Closing dataloader and tfboard if used")
        if args.use_tfboard and not args.no_save:
            tblogger_gen.close()
            tblogger_dis_real.close()
            tblogger_dis_fake.close()
        logger.info('Aborted training.')
        return

    logger.info('Finished training.')
    time.sleep(5) # sleep some time to make sure that cache is free for testing
    logger.info("Start testing final model")

    test_output_dir = os.path.join(output_dir, 'testing')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    if final_model is not None:
        if args.multi_gpu_testing:
            args_test = Namespace(cfg_file='{}'.format(args.cfg_file),
                                  load_ckpt='{}'.format(final_model),
                                  load_dis=None, load_gen=None,
                                  multi_gpu_testing=True, output_dir='{}'.format(test_output_dir),
                                  range=None, set_cfgs=args.set_cfgs, vis=False)
        else:
            args_test = Namespace(cfg_file='{}'.format(args.cfg_file),
                                  load_ckpt='{}'.format(final_model),
                                  load_dis=None, load_gen=None,
                                  multi_gpu_testing=False, output_dir='{}'.format(test_output_dir),
                                  range=None, set_cfgs=args.set_cfgs, vis=False)

        test_net_routine(args_test)


if __name__ == '__main__':
    main()
