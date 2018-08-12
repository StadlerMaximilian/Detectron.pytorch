import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from core.config import cfg
import utils.net as net_utils


class VGG_CNN_M_1024_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 7, padding=0, stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.LocalResponseNorm(size=5, alpha=0.0005, beta=0.75, k=2.),
                                   nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, padding=0, stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.LocalResponseNorm(size=5, alpha=0.0005, beta=0.75, k=2.),
                                   nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
                                  )

        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1, stride=1),
                                   nn.ReLU(inplace=True),
                                   )

        self.spatial_scale = 1. / 16.
        self.spatial_scale_base = 1. / 2.
        self.dim_out_base = 96
        self.dim_out = 512
        self.resolution = 6

        # freeze gradients for first bottom convolutional blocks
        freeze_params(self.conv1)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for block_id in range(5):
            block_name = 'conv{}'.format(block_id + 1)
            torch_name = block_name + '.0.' #as conv layer is always first in every block
            caffe_name = 'conv{}_'.format(block_id + 1)
            mapping_to_detectron[torch_name + 'weight'] = caffe_name + 'w'
            mapping_to_detectron[torch_name + 'bias'] = caffe_name + 'b'
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x, req_fake_features=False):
        x = self.conv1(x)

        if cfg.GAN.GAN_MODE_ON and req_fake_features:
            x_base = x.clone()

        for i in range(1, 5):
            x = getattr(self, 'conv{}'.format(i+1))(x)

        if cfg.GAN.GAN_MODE_ON and req_fake_features:
            return x, x_base
        else:
            return x


class VGG_CNN_M_1024_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, resolution=6):
        super().__init__()
        self.roi_xform = net_utils.roiPoolingLayer(roi_xform_func, spatial_scale, resolution)

        self.fc1 = nn.Linear(dim_in * resolution * resolution, 4096)

        self.fc2 = nn.Linear(4096, 1024)
        self.dim_out = 1024

        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(x, rpn_ret)

        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
