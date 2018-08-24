import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import importlib

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import utils.net as net_utils
import utils.blob as blob_utils


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        raise


class Generator(nn.Module):
    """
    Class representing the Generator of the Perceptual GAN
    it includes functionality of Conv_Body with RPN (with is assumed to be fixed during training Perceptual GAN), the
    usage of the branch of conv1 features, additional pooling and is building the main GeneratorBlock upon this pooled
    features
    """
    def __init__(self, pretrained_weights=None):
        assert cfg.GAN.GAN_MODE_ON
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.provide_fake_features = True

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.Conv_Body._set_provide_fake_features(True)

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        self.roi_pool = net_utils.roiPoolingLayer(self.roi_feature_transform, self.Conv_Body.spatial_scale,
                                                  self.Conv_Body.resolution)

        self.Generator_Block = GeneratorBlock(self.roi_feature_transform, self.Conv_Body.spatial_scale_base,
                                              self.Conv_Body.resolution, self.Conv_Body.dim_out_base,
                                              self.Conv_Body.dim_out)

        self._init_modules(pretrained_weights)

    def _init_modules(self, pretrained_weights=None):
        """
        inits layers and loads pretrained detectron-backbone-architecture (Conv_Body and RPN)
        also freezes weights of Conv_Body and RPN
        """
        if pretrained_weights is None:
            return

        pretrained_detectron = torch.load(pretrained_weights)
        load_layers = ['Conv_Body', 'RPN']
        mapping, _ = self.detectron_weight_mapping()
        state_dict = {}
        ckpt = pretrained_detectron['model']
        for name in ckpt:
            if name.split('.')[0] in load_layers:
                if mapping[name]:
                    state_dict[name] = ckpt[name]
        self.load_state_dict(state_dict, strict=False)
        del pretrained_detectron
        torch.cuda.empty_cache()

        # finaly freeze Conv_Body and RPN
        freeze_params(self.Conv_Body)
        freeze_params(self.RPN)

    def forward(self, data, im_info, roidb=None, mode="FAKE", train_part="GENERATOR", **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        if self.provide_fake_features:
            blob_conv, blob_conv_base = self.Conv_Body(im_data)
        else:
            blob_conv = self.Conv_Body(im_data)

        return_dict['blob_conv'] = blob_conv

        rpn_ret = self.RPN(blob_conv, im_info, roidb, mode=mode, train_part=train_part)
        return_dict['rpn_ret'] = rpn_ret

        blob_conv_pooled = self.roi_pool(blob_conv, rpn_ret)
        return_dict['blob_conv_pooled'] = blob_conv_pooled

        if self.provide_fake_features:
            blob_conv_residual = self.Generator_Block(blob_conv_base, rpn_ret)
            return_dict['bloc_conv_residual'] = blob_conv_residual
            return_dict['blob_fake'] = blob_conv_pooled + blob_conv_residual

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

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
        self.Conv_Body._set_provide_fake_features(bool)


########################################################################################################################


class ResidualBlock(nn.Module):
    """
    Class representing the ResidualBlock used in the generator
    """
    def __init__(self, in_channels=512, num=512, kernel=3, stride=1):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.block = nn.Sequential(nn.Conv2d(in_channels, num, kernel, stride=stride, padding=1),
                                   nn.BatchNorm2d(num),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num, num, kernel, stride=stride, padding=1),
                                   nn.BatchNorm2d(num)
                                   )
        self._init_weights()

    def forward(self, x):
        y = self.block(x)
        return y + x

    def detectron_weight_mapping(self):
        """ mapping only for being able to load models correctly, original detectron not supported"""
        detectron_weight_mapping = {
            'block.0.weight': 'blockConv1_w',
            'block.0.bias': 'blockConv2_b',
            'block.1.weight': 'blockBN1_w',
            'block.1.bias': 'blockBN1_b',
            'block.3.weight': 'blockConv2_w',
            'block.3.bias': 'blockConv2_b',
            'block.4.weight': 'blockBN2_w',
            'block.4.bias': 'blockBN2_b',
        }
        orphan_in_detectron = []
        self.mapping_to_detectron = detectron_weight_mapping
        self.orphans_in_detectron = orphan_in_detectron
        return self.mapping_to_detectron, self.orphans_in_detectron

    def _init_weights(self):
        init.kaiming_uniform_(self.block[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.block[0].bias, 0)
        init.kaiming_uniform_(self.block[3].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.block[3].bias, 0)

########################################################################################################################


class GeneratorBlock(nn.Module):
    """
    Class representing the actual Generator architecture
    """
    def __init__(self, roi_xform_func, spatial_scale_base, resolution, dim_out_base, dim_out):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.gen_base = nn.Sequential(nn.Conv2d(dim_out_base, 256, 3, padding=1, stride=2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, dim_out, 1, padding=0, stride=1),
                                      nn.ReLU(inplace=True)
                                      )
        self._init_weights()

        self.spatial_scale = spatial_scale_base * (1. / 2.)

        self.roi_xform = net_utils.roiPoolingLayer(roi_xform_func, self.spatial_scale, resolution)

        for n in range(cfg.GAN.MODEL.NUM_BLOCKS):
            self.add_module('gen_res_block' + str(n + 1), ResidualBlock(in_channels=dim_out, num=dim_out))

    def _init_weights(self):
        init.kaiming_uniform_(self.gen_base[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.gen_base[0].bias, 0)
        init.kaiming_uniform_(self.gen_base[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.gen_base[2].bias, 0)

    def forward(self, x_base, rpn_ret):
        x = self.gen_base(x_base)
        x = self.roi_xform.forward(x, rpn_ret)

        for n in range(cfg.GAN.MODEL.NUM_BLOCKS):
            x = self.__getattr__('gen_res_block' + str(n+1))(x)

        return x

    def detectron_weight_mapping(self):
        """ mapping only for being able to load models correctly, original detectron not supported"""
        d_wmap = {}
        d_orphan = []
        d_wmap['gen_base.0.weight'] = 'baseConv1_w'
        d_wmap['gen_base.0.bias'] = 'baseConv1_b'
        d_wmap['gen_base.2.weight'] = 'baseConv2_w'
        d_wmap['gen_base.2.bias'] = 'baseConv2_b'

        for name, m_child in self.named_children():
            if name == 'gen_base': #skip gen_base
                continue
            if list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                d_orphan.extend(child_orphan)
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = d_orphan
        return self.mapping_to_detectron, self.orphans_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False

