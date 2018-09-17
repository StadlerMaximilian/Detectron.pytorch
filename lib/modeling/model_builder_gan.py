import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from modeling.generator import Generator
from modeling.discriminator import Discriminator
from core.config import cfg

from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import utils.net as net_utils


class GAN(nn.Module):
    def __init__(self, generator_weights=None, discriminator_weights=None):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.generator = Generator(self.roi_feature_transform, generator_weights)
        resolution = self.generator.Conv_Body.resolution

        if cfg.RPN.RPN_ON:
            dim_in = self.generator.RPN.dim_out
        else:
            dim_in = self.generator.Conv_Body.dim_out

        self.discriminator = Discriminator(dim_in, resolution, discriminator_weights)

    def forward(self, data, im_info, roidb=None, flags=None, adv_target=None, **rpn_kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(data, im_info, roidb, flags, adv_target, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, flags=None, adv_target=None, **rpn_kwargs):

        outputs_gen = self.generator(data, im_info, roidb, flags, **rpn_kwargs)

        if self.training:
            rpn_ret = outputs_gen['rpn_ret']

            input_discriminator = {'blob_conv': outputs_gen['blob_conv'],
                                   'rpn_ret': rpn_ret,
                                   'adv_target': adv_target,
                                   'flags': flags
                                   }
        else:
            rpn_ret = outputs_gen['rpn_ret']
            input_discriminator = {'blob_conv': outputs_gen['blob_fake'],
                                   'rpn_ret': rpn_ret
                                   }

        dis_out = self.discriminator(**input_discriminator)

        if not self.training: # if eval only
            copy_blobs = ['blob_conv_pooled', 'blob_fake', 'blob_conv_residual', 'rpn_ret']
            for key in copy_blobs:
                dis_out[key] = outputs_gen[key]

        #  if cfg.DEBUG:
        #      print("\t memory: allocated: {} (max: {}), cached: {} (max: {})".format(torch.cuda.memory_allocated(),
        #                                                                              torch.cuda.max_memory_allocated(),
        #                                                                              torch.cuda.memory_cached(),
        #                                                                               torch.cuda.max_memory_cached()))

        return dis_out

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

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """

        if cfg.DEBUG:
            print("roi_feature_transform: resolution: {}, spatial_scale: {}".format(resolution, spatial_scale))

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