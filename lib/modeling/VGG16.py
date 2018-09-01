import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils


class VGG16_conv5_body():
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1, stride=1),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2, stride=2)
                                  )

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   )

        self.spatial_scale = 1. / 16.
        self.spatial_scale_base = 1. / 2.
        self.dim_out_base = 96
        self.dim_out = 512
        self.resolution = 7

        # freeze gradients for first two bottom convolutional blocks
        freeze_params(self.conv1)
        freeze_params(self.conv2)

    def detectron_weight_mapping(self):
        blocks = [2, 2, 3, 3, 3]
        conv_ids = [[0, 2], [0, 2], [0, 2, 4], [0, 2, 4], [0, 2, 4]]

        mapping_to_detectron = {}
        for block_id in range(5):
            block_name = 'conv{}'.format(block_id + 1)
            for layer_id in range(blocks[block_id]):
                torch_name = block_name + '.' + str(conv_ids[layer_id]) + '.'
                caffe_name = 'conv{}_{}.'.format(block_id, layer_id)
                mapping_to_detectron[torch_name + 'weight'] = caffe_name + 'w'
                mapping_to_detectron[torch_name + 'bias'] = caffe_name + 'b'
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        x = self.conv1(x)

        if cfg.GAN.GAN_MODE_ON:
            x_base = x.clone()

        for i in range(1, 5):
            x = getattr(self, 'conv{}'.format(i+1))(x)

        if cfg.GAN.GAN_MODE_ON:
            return x, x_base
        else:
            return x


class VGG16_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, resolution=7):
        super().__init__()
        self.roi_xform = net_utils.roiPoolingLayer(roi_xform_func, spatial_scale, resolution)

        self.fc1 = nn.Sequential(nn.Linear(dim_in * resolution * resolution, 4096),
                                 nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.ReLU(inplace=True))
        self.dim_out = 4096

    def _init_weights(self):
        init.kaiming_uniform_(self.fc1[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1[0].bias, 0)
        init.kaiming_uniform_(self.fc2[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc2[0].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.0.weight': 'fc6_w',
            'fc1.0.bias': 'fc6_b',
            'fc2.0.weight': 'fc7_w',
            'fc2.0.bias': 'fc7_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(x, rpn_ret)

        batch_size = x.size(0)
        x = self.fc1(x.view(batch_size, -1))
        x = self.fc2(x)

        return x


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


