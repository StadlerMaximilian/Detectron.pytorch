import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import os

from core.config import cfg
from modeling.model_builder import get_func, compare_state_dict, check_inference, Generalized_RCNN
from modeling.generator import Generator
from modeling.discriminator import Discriminator
import nn as mynn
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.detectron_weight_helper as weight_utils
import utils.net as net_utils


class GAN(nn.Module):
    def __init__(self, generator_weights=None, discriminator_weights=None):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.generator = Generator()
        resolution = self.generator.Conv_Body.resolution
        dim_in = self.generator.RPN.dim_out
        self.discriminator = Discriminator(dim_in, resolution)
        self.provide_fake_features = True

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):

        gen_out = self.generator(data, im_info, roidb, **rpn_kwargs)

        if self.provide_fake_features:
            blob_conv = gen_out['blob_fake']
        else:
            blob_conv = gen_out['blob_conv']
        rpn_ret = gen_out['rpn_ret']

        dis_out = self.discriminator(blob_conv, rpn_ret)

        return dis_out

    def _init_module(self, generator_weights=None, discriminator_weights=None):
        if generator_weights is None or discriminator_weights is None:
            return
        else:
            pretrained_generator = torch.load(generator_weights)
            pretrained_discriminator = torch.load(discriminator_weights)

            ckpt = pretrained_discriminator['model']
            state_dict = {}
            for name in ckpt:
                state_dict[name] = ckpt[name]
            self.discriminator.load_state_dict(state_dict, strict=False)

            ckpt = pretrained_generator['model']
            state_dict = {}
            for name in ckpt:
                state_dict[name] = ckpt[name]
            self.generator.load_state_dict(state_dict, strict=False)

            del pretrained_discriminator
            del pretrained_generator
            torch.cuda.empty_cache()

    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _set_provide_fake_features(self, bool):
        self.provide_fake_features = bool
        self.generator.set_provide_fake_features(bool)