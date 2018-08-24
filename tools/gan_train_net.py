""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

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
    collate_minibatch_discriminator, collate_minibatch_generator
from modeling.generator import Generator
from modeling.discriminator import Discriminator
from modeling.model_builder_gan import GAN
from utils.logging import setup_logging
from utils.timer import Timer
from utils.gan_utils import TrainingStats, ModeFlags
from test_net import test_net_routine

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
        required=True
    )

    parser.add_argument(
        '--load_ckpt_G', help='checkpoint path of Generator to load')
    parser.add_argument(
        '--load_ckpt_D', help='checkpoint path of Discriminator to load')

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
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    return parser.parse_args()


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
    model_state_dict = model.state_dict()
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
    })
    logger.info('save model: %s', save_name)
    return save_name


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

    if args.load_pretrained is None:
        raise ValueError("No pretrained detectron model specified")
    else:
        if not os.path.exists(args.load_pretrained):
            raise ValueError("Specified pretrained detectron model does not exists")
        else:
            cfg.GAN.TRAIN.PRETRAINED_WEIGHTS = args.load_pretrained

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # Adaptively adjust some configs for discriminator #
    original_batch_size_D = cfg.NUM_GPUS * cfg.GAN.TRAIN.IMS_PER_BATCH_D
    original_ims_per_batch_D = cfg.GAN.TRAIN.IMS_PER_BATCH_D
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size_D is None:
        args.batch_size_D = original_batch_size_D
    cfg.NUM_GPUS = torch.cuda.device_count()
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
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size_G is None:
        args.batch_size_G = original_batch_size_G
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size_G % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size_G, cfg.NUM_GPUS)
    cfg.GAN.TRAIN.IMS_PER_BATCH_G = args.batch_size_G // cfg.NUM_GPUS
    effective_batch_size_G = args.iter_size * args.batch_size_G
    print('effective_batch_size_D = batch_size * iter_size = %d * %d' % (args.batch_size_G, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size_G, effective_batch_size_G))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch_G, cfg.GAN.TRAIN.IMS_PER_BATCH_G))

    # Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr_D = cfg.GAN.SOLVER.BASE_LR_D
    old_base_lr_G = cfg.GAN.SOLVER.BASE_LR_G
    cfg.GAN.SOLVER.BASE_LR_D *= args.batch_size_D / original_batch_size_D
    cfg.GAN.SOLVER.BASE_LR_G *= args.batch_size_G / original_batch_size_G
    print('Adjust BASE_LR_D linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr_D, cfg.GAN.SOLVER.BASE_LR_D))
    print('Adjust BASE_LR_G linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr_G, cfg.GAN.SOLVER.BASE_LR_G))

    # Adjust solver steps
    step_scale_D = original_batch_size_D / effective_batch_size_D
    step_scale_G = original_batch_size_G / effective_batch_size_G
    if not cfg.GAN.SOLVER.STEPS_D:
        cfg.GAN.SOLVER.STEPS_D = cfg.GAN.SOLVER.STEPS
    if not cfg.GAN.SOLVER.STEPS_G:
        cfg.GAN.SOLVER.STEPS_G = cfg.GAN.SOLVER.STEPS
    old_solver_steps_D = cfg.GAN.SOLVER.STEPS_D
    old_solver_steps_G = cfg.GAN.SOLVER.STEPS_G
    old_max_iter = cfg.GAN.SOLVER.MAX_ITER
    cfg.GAN.SOLVER.STEPS_D = list(map(lambda x: int(x * step_scale_D + 0.5), cfg.GAN.SOLVER.STEPS_D))
    cfg.GAN.SOLVER.STEPS_G = list(map(lambda x: int(x * step_scale_G + 0.5), cfg.GAN.SOLVER.STEPS_G))
    cfg.GAN.SOLVER.MAX_ITER_D = int(cfg.GAN.SOLVER.MAX_ITER * step_scale_D + 0.5)
    cfg.GAN.SOLVER.MAX_ITER_G = int(cfg.GAN.SOLVER.MAX_ITER * step_scale_G + 0.5)
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

    num_loaders = 3
    train_size_D = 0
    train_size_G = 0
    train_size = 0

    # Dataset #
    timers['roidb_source'].tic()
    roidb_source, ratio_list_source, ratio_index_source = combined_roidb_for_training(
        cfg.GAN.TRAIN.DATASETS_SOURCE, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb_source'].toc()
    roidb_size_source = len(roidb_source)
    logger.info('{:d} roidb entries'.format(roidb_size_source))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb_source'].average_time)

    # Effective training sample size for one epoch
    train_size_D += roidb_size_source // args.batch_size_D * args.batch_size_D

    batchSampler_source_discriminator= BatchSampler(
        sampler=MinibatchSampler(ratio_list_source, ratio_index_source, cfg.GAN.TRAIN.IMS_PER_BATCH_D),
        batch_size=args.batch_size_D,
        drop_last=True
    )

    dataset_source_discriminator = RoiDataLoader(
        roidb_source,
        cfg.MODEL.NUM_CLASSES,
        training=True)

    dataloader_source_discriminator = torch.utils.data.DataLoader(
        dataset_source_discriminator,
        batch_sampler=batchSampler_source_discriminator,
        num_workers=int(cfg.DATA_LOADER.NUM_THREADS / num_loaders),
        collate_fn=collate_minibatch_discriminator,
        pin_memory=False)

    dataiterator_source_discriminator = iter(dataloader_source_discriminator)

    timers['roidb_target'].tic()
    roidb_target, ratio_list_target, ratio_index_target = combined_roidb_for_training(
        cfg.GAN.TRAIN.DATASETS_TARGET, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb_target'].toc()
    roidb_size_target = len(roidb_target)
    logger.info('{:d} roidb entries'.format(roidb_size_target))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb_target'].average_time)

    # Effective training sample size for one epoch
    train_size_D += roidb_size_target // args.batch_size_D * args.batch_size_D

    batchSampler_target_discriminator = BatchSampler(
        sampler=MinibatchSampler(ratio_list_target, ratio_index_target, cfg.GAN.TRAIN.IMS_PER_BATCH_D),
        batch_size=args.batch_size_D,
        drop_last=True
    )

    dataset_target_discriminator = RoiDataLoader(
        roidb_target,
        cfg.MODEL.NUM_CLASSES,
        training=True)

    dataloader_target_discriminator = torch.utils.data.DataLoader(
        dataset_target_discriminator,
        batch_sampler=batchSampler_target_discriminator,
        num_workers=int(cfg.DATA_LOADER.NUM_THREADS / num_loaders),
        collate_fn=collate_minibatch_discriminator,
        pin_memory=False)

    dataiterator_target_discriminator = iter(dataloader_target_discriminator)

    timers['roidb_target_g'].tic()
    roidb_target_g, ratio_list_target_g, ratio_index_target_g = combined_roidb_for_training(
        cfg.GAN.TRAIN.DATASETS_TARGET, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb_target_g'].toc()
    roidb_size_target_g = len(roidb_target_g)
    logger.info('{:d} roidb entries'.format(roidb_size_target_g))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb_target_g'].average_time)

    # Effective training sample size for one epoch
    train_size_G += roidb_size_target_g // args.batch_size_G * args.batch_size_G

    batchSampler_target_generator = BatchSampler(
        sampler=MinibatchSampler(ratio_list_target_g, ratio_index_target_g, cfg.GAN.TRAIN.IMS_PER_BATCH_G),
        batch_size=args.batch_size_G,
        drop_last=True
    )

    dataset_target_generator = RoiDataLoader(
        roidb_target_g,
        cfg.MODEL.NUM_CLASSES,
        training=True)

    dataloader_target_generator = torch.utils.data.DataLoader(
        dataset_target_generator,
        batch_sampler=batchSampler_target_generator,
        num_workers=int(cfg.DATA_LOADER.NUM_THREADS / num_loaders),
        collate_fn=collate_minibatch_generator,
        pin_memory=False)

    dataiterator_target_generator = iter(dataloader_target_generator)
    train_size = max(train_size_D // 2, train_size_G)

    # Model
    generator = Generator(pretrained_weights=cfg.GAN.TRAIN.PRETRAINED_WEIGHTS) # pretrained_weights
    resolution = generator.Conv_Body.resolution
    dim_in = generator.RPN.dim_out
    discriminator = Discriminator(dim_in, resolution, pretrained_weights=cfg.GAN.TRAIN.PRETRAINED_WEIGHTS)

    if cfg.CUDA:
        generator.cuda()
        discriminator.cuda()

    # Discriminator Parameters
    params = {}
    params['D'] = {
        'bias_params': [],
        'bias_param_names': [],
        'nonbias_params': [],
        'nonbias_param_names': [],
        'nograd_param_names': []
    }

    for key, value in discriminator.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params['D']['bias_params'].append(value)
                params['D']['bias_param_names'].append(key)
            else:
                params['D']['nonbias_params'].append(value)
                params['D']['nonbias_param_names'].append(key)
        else:
            params['D']['nograd_param_names'].append(key)

    params_D = [
        {'params': params['D']['nonbias_params'],
         'lr': 0,
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_D},
        {'params': params['D']['bias_params'],
         'lr': 0 * (cfg.GAN.SOLVER.BIAS_DOUBLE_LR_D + 1),
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_D if cfg.GAN.SOLVER.BIAS_WEIGHT_DECAY_D else 0}
    ]
    # names of paramerters for each paramter
    param_names_D = [params['D']['nonbias_param_names'], params['D']['bias_param_names']]

    ### Generator Parameters ###
    params['G'] = {
        'bias_params': [],
        'bias_param_names': [],
        'nonbias_params': [],
        'nonbias_param_names': [],
        'nograd_param_names': []
    }

    for key, value in generator.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params['G']['bias_params'].append(value)
                params['G']['bias_param_names'].append(key)
            else:
                params['G']['nonbias_params'].append(value)
                params['G']['nonbias_param_names'].append(key)
        else:
            params['G']['nograd_param_names'].append(key)

    params_G = [
        {'params': params['G']['nonbias_params'],
         'lr': 0,
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_G},
        {'params': params['G']['bias_params'],
         'lr': 0 * (cfg.GAN.SOLVER.BIAS_DOUBLE_LR_G + 1),
         'weight_decay': cfg.GAN.SOLVER.WEIGHT_DECAY_G if cfg.GAN.SOLVER.BIAS_WEIGHT_DECAY_G else 0}
    ]
    # names of parameters for each parameter
    param_names_G = [params['G']['nonbias_param_names'], params['G']['bias_param_names']]

    ### Optimizers ###
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

    optimizer_D.zero_grad()
    optimizer_G.zero_grad()

    ### Load checkpoint
    if args.load_ckpt_G and args.load_ckpt_D:
        load_name_G = args.load_ckpt_G
        load_name_D = args.load_ckpt_D
        logging.info("loading checkpoint %s", load_name_G)
        logging.info("loading checkpoint %s", load_name_D)
        checkpoint_G = torch.load(load_name_G, map_location=lambda storage, loc: storage)
        checkpoint_D = torch.load(load_name_D, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(generator, checkpoint_G['model'])
        net_utils.load_ckpt(discriminator, checkpoint_D['model'])

        if args.resume:
            # as for every k steps of discriminator, G is updated once
            optimizer_G.load_state_dict(checkpoint_G['optimizer'])
            optimizer_D.load_state_dict(checkpoint_D['optimizer'])
        del checkpoint_G
        del checkpoint_D
        torch.cuda.empty_cache()

    lr_D = optimizer_D.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.
    lr_G = optimizer_G.param_groups[0]['lr']

    generator = mynn.DataParallel(generator, cpu_keywords=['im_info', 'roidb'],
                                  minibatch=True, batch_outputs=False) # keep batch split onto GPUs for generator
    discriminator = mynn.DataParallel(discriminator, cpu_keywords=['im_info', 'roidb'],
                                      minibatch=True)

    ### Training Setups ###
    args.run_name = misc_utils.get_run_name() + '_step'
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    output_dir_D = misc_utils.get_output_dir_part(args, args.run_name, 'discriminator')
    output_dir_G = misc_utils.get_output_dir_part(args, args.run_name, 'generator')
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir_D):
            os.makedirs(output_dir_D)
        if not os.path.exists(output_dir_G):
            os.makedirs(output_dir_G)
        logging.info("Using output_dirs: {}\n\t\t{}\n\t\t{}".format(output_dir, output_dir_G,
                                                                    output_dir_D))

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    generator.train()
    discriminator.train()
    fake_dis_flags = [ModeFlags("fake", "discriminator") for _ in range(cfg.NUM_GPUS)]
    real_dis_flags = [ModeFlags("real", "discriminator") for _ in range(cfg.NUM_GPUS)]
    fake_gen_flags = [ModeFlags("fake", "generator") for _ in range(cfg.NUM_GPUS)]
    # use smoothed label for "REAL" - Label
    adv_target_smoothed = [cfg.GAN.MODEL.LABEL_SMOOTHING] * cfg.NUM_GPUS
    # 0.0 for fake in discriminator
    adv_target_zero = [0.0] * cfg.NUM_GPUS

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    # Set index for decay steps
    decay_steps_ind_D = None
    decay_steps_ind_G = None
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

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)

    # use maximum max_iter for training
    max_iter = max(cfg.GAN.SOLVER.MAX_ITER_D, cfg.GAN.SOLVER.MAX_ITER_G)

    try:
        logger.info('Training starts !')
        step = args.start_step
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

            training_stats.IterTic()

            # train discriminator
            for _ in range(cfg.GAN.TRAIN.k):

                optimizer_D.zero_grad()

                mem = torch.cuda.max_memory_allocated()
                print("Training D1 with mem: {}".format(mem))

                # train on fake data
                try:
                    input_data_fake = next(dataiterator_target_discriminator)
                except StopIteration:
                    dataiterator_target_discriminator = iter(dataloader_target_discriminator)
                    input_data_fake = next(dataiterator_target_discriminator)

                for key in input_data_fake:
                    if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                        input_data_fake[key] = list(map(Variable, input_data_fake[key]))

                generator.module._set_provide_fake_features(True)
                input_data_fake.update({"flags": fake_dis_flags})
                outputs_G_fake = generator(**input_data_fake)
                blob_fake = [x['blob_fake'] for x in outputs_G_fake]
                rpn_ret_fake = [x['rpn_ret'] for x in outputs_G_fake]
                input_discriminator = {'blob_conv': blob_fake,
                                       'rpn_ret': rpn_ret_fake,
                                       'adv_target': adv_target_zero
                                       }
                outputs_D_fake = discriminator(**input_discriminator)
                training_stats.UpdateIterStats(out_D_fake=outputs_D_fake)
                loss_D_fake = outputs_D_fake['total_loss']

                mem = torch.cuda.max_memory_allocated()
                print("Finished training D1 with mem: {}".format(mem))

                mem = torch.cuda.max_memory_allocated()
                print("Training D2 with mem: {}".format(mem))

                # train on real data
                try:
                    input_data_real = next(dataiterator_source_discriminator)
                except StopIteration:
                    dataiterator_source_discriminator = iter(dataloader_source_discriminator)
                    input_data_real = next(dataiterator_source_discriminator)

                for key in input_data_real:
                    if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                        input_data_real[key] = list(map(Variable, input_data_real[key]))

                generator.module._set_provide_fake_features(False)
                input_data_real.update({"flags": real_dis_flags})
                outputs_G_real = generator(**input_data_real)
                blob_conv_pooled = [x['blob_conv_pooled'] for x in outputs_G_real]
                rpn_ret_real = [x['rpn_ret'] for x in outputs_G_real]
                input_discriminator = {'blob_conv': blob_conv_pooled,
                                       'rpn_ret': rpn_ret_real,
                                       'adv_target': adv_target_smoothed
                                       }
                outputs_D_real = discriminator(**input_discriminator)
                training_stats.UpdateIterStats(out_D_real=outputs_D_real)
                loss_D_real = outputs_D_real['total_loss']

                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()

                mem = torch.cuda.max_memory_allocated()
                print("Finished training D2 with mem: {}".format(mem))

                mem = torch.cuda.max_memory_allocated()
                print("Training G with mem: {}".format(mem))

            # train generator
            optimizer_G.zero_grad()

            try:
                input_data_fake = next(dataiterator_target_generator)
            except StopIteration:
                dataiterator_target_generator = iter(dataloader_target_generator)
                input_data_fake = next(dataiterator_target_generator)

            for key in input_data_fake:
                if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                    input_data_fake[key] = list(map(Variable, input_data_fake[key]))

            generator.module._set_provide_fake_features(True)
            input_data_fake.update({"flags": fake_gen_flags})
            outputs_GG = generator(**input_data_fake)
            blob_fake_g = [x['blob_fake'] for x in outputs_GG]
            rpn_ret_g = [x['rpn_ret'] for x in outputs_GG]
            # also use smoothed value for GENERATOR training
            input_discriminator = {'blob_conv': blob_fake_g,
                                   'rpn_ret': rpn_ret_g,
                                   'adv_target': adv_target_smoothed
                                   }
            outputs_DG = discriminator(**input_discriminator)
            training_stats.UpdateIterStats(out_G=outputs_DG)

            loss_G = outputs_DG['total_loss']
            loss_G.backward()
            optimizer_G.step()

            mem = torch.cuda.max_memory_allocated()
            print("Finished training G with mem: {}".format(mem))

            training_stats.IterToc()

            training_stats.LogIterStats(step, lr_D=lr_D, lr_G=lr_G)

            # free cuda cache
            torch.cuda.empty_cache()

            mem = torch.cuda.max_memory_allocated()
            print("Freed cache: mem: {}".format(mem))

            if (step+1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(output_dir_G, args, step, train_size, generator, optimizer_G, "G")
                save_ckpt(output_dir_D, args, step, train_size, discriminator, optimizer_D, "D")

        # ---- Training ends ----
        # Save last checkpoint
        final_generator = save_ckpt(output_dir_G, args, step, train_size, generator, optimizer_G, "G")
        final_discriminator = save_ckpt(output_dir_D, args, step, train_size, discriminator, optimizer_D, "D")

        gan = GAN(generator_weights=final_generator, discriminator_weights=final_discriminator)
        final_model = save_model(output_dir, no_save=False, model=gan)

    except (RuntimeError, KeyboardInterrupt):

        del dataiterator_source_discriminator
        del dataiterator_target_discriminator
        del dataiterator_target_generator

        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir_G, args, step, train_size, generator, optimizer_G, "G")
        save_ckpt(output_dir_D, args, step, train_size, discriminator, optimizer_D, "D")
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)
        logger.info("Closing dataloader and tfboard if used")
        if args.use_tfboard and not args.no_save:
            tblogger.close()
        logger.info('Aborted training.')
        return

    logger.info("Closing dataloader and tfboard if used")
    if args.use_tfboard and not args.no_save:
        tblogger.close()
    logger.info('Finished training.')

    logger.info("Start testing final model")

    if final_model is not None:
        args_test = Namespace(cfg_file='{}'.format(args.cfg_file), dataset=None,
                              load_ckpt='{}'.format(final_model), load_detectron=None,
                              multi_gpu_testing=True, output_dir='{}'.format(cfg.OUTPUT_DIR),
                              range=None, set_cfgs=[], vis=False)
        test_net_routine(args_test)


if __name__ == '__main__':
    main()
