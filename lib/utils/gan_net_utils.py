import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import utils.net as net_utils


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=512, num=512, kernel=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, num, kernel, stride=stride, padding=1),
                                   nn.BatchNorm2d(num),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num, num, kernel, stride=stride, padding=1),
                                   nn.BatchNorm2d(num)
                                   )

    def forward(self, x):
        y = self.block(x)
        return y + x


class GeneratorBlock(nn.Module):
    def __init__(self, roi_xform_func, spatial_scale_base, resolution, dim_out_base, dim_out):
        super().__init__()

        self.gen_base = nn.Sequential(nn.Conv2d(dim_out_base, 256, 3, padding=1, stride=2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, dim_out, 1, padding=0, stride=1),
                                      nn.ReLU(inplace=True)
                                      )

        self.spatial_scale = spatial_scale_base * (1. / 2.)

        self.roi_xform = net_utils.roiPoolingLayer(roi_xform_func, self.spatial_scale, resolution)

        for n in range(cfg.GAN.MODEL.NUM_BLOCKS):
            self.add_module('res_block' + str(n + 1), ResidualBlock(in_channels=512, num=512))

    def forward(self, x_base, rpn_ret):
        x = self.gen_base(x_base)
        x = self.roi_xform.forward(x, rpn_ret)

        for n in range(cfg.GAN.MODEL.NUM_BLOCKS):
            x = self.__getattr__('res_block' + str(n+1))(x)

        return x
