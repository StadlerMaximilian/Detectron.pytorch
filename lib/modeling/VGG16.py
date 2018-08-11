import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils


class VGG16_conv5_body():
    def __init__(self, block_counts):
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
        self.dim_out = 512

        # freeze gradients for first two bottom convolutional blocks
        freeze_params(self.conv1)
        freeze_params(self.conv2)

    def detectron_weight_mapping(self):
        blocks = [2, 2, 3, 3, 3]
        mapping_to_detectron = {}
        for block_id in range(5):
            block_name = 'conv{}'.format(block_id + 1)
            for layer_id in range(blocks[block_id]):
                layer_name = 'conv{}'.format(layer_id + 1)
                torch_name = block_name + '.' + layer_name + '.'
                caffe_name = 'conv{}_{}.'.format(block_id, layer_id)
                mapping_to_detectron[torch_name + 'weight'] = caffe_name + 'w'
                mapping_to_detectron[torch_name + 'bias'] = caffe_name + 'b'
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        for i in range(5):
            x = getattr(self, 'conv{}'.format(i+1))(x)
        return x


class VGG16_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        self.fc6 = nn.Sequential(nn.Linear(dim_in * 7 * 7, 4096),
                                 nn.ReLU(inplace=True)
                                 )
        self.fc7 = nn.Linear(4096, 4096)
        self.dim_out = 4096

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc6.linear.weight': 'fc6_w',
            'fc6.linear.bias': 'fc6_b',
            'fc7.linear.weight': 'fc7_w',
            'fc7.linear.bias': 'fc7_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=7,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.fc6(x)
        x = self.fc7(x)

        return x


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


