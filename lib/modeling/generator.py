import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
from modeling.model_builder import get_func
import modeling.rpn_heads as rpn_heads
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


class Generator(nn.Module):
    def __init__(self):
        assert cfg.GAN.GAN_MODE_ON
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        self.Pooled_Features = get_func(cfg.GAN.MODEL.CONV_BODY_ROI_POOLING)(self.roi_feature_transform,
                                                                             self.Conv_Body.spatial_scale,
                                                                             self.Conv_Body.resolution)

        self.Generator_Block = GeneratorBlock(self.roi_feature_transform, self.Conv_Body.spatial_scale_base,
                                                        self.Conv_Body.resolution, self.Conv_Body.dim_out_base,
                                                        self.Conv_Body.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_PRETRAINED_DETECTRON_WEIGHTS:
            weight_utils.load_caffe2_detectron_weights(self, cfg.MODEL.PRETRAINED_BACKBONE_WEIGHTS)

    def forward(self, data, im_info, req_fake_features=True, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        if req_fake_features:
            blob_conv, blob_conv_base = self.Conv_Body(im_data, req_fake_features=True)
        else:
            blob_conv = self.Conv_Body(im_data, req_fake_features=True)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        self.Pooled_Features

        if req_fake_features:
            blob_conv_fake = self.Generator_Block(blob_conv_base, rpn_ret)

        if not self.training:
            return_dict['blob_conv'] = blob_conv
            if req_fake_features:
                return_dict['bloc_conv_fake'] = blob_conv_fake

        return return_dict
