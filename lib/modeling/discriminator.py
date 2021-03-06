import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

from core.config import cfg
from modeling.model_builder import get_func, compare_state_dict, check_inference, Generalized_RCNN
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.detectron_weight_helper as weight_utils
import utils.net as net_utils
import nn as mynn


class Discriminator(nn.Module):
    def __init__(self, dim_in, resolution, pretrained_weights=None):
        super().__init__()
        self.fc_dim = dim_in * resolution * resolution
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.adversarial = nn.Sequential(nn.Linear(self.fc_dim, 4096),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(4096, 1024),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Linear(1024, 1),
                                         nn.Sigmoid())

        self.adversarial_criterion = nn.BCELoss()

        self._init_weights()

        self.Box_Head = get_func(cfg.GAN.MODEL.CONV_BODY_FC_HEAD)(dim_in, resolution)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)
        self._init_modules(pretrained_weights)

    def forward(self, blob_conv, rpn_ret, adv_target=None, flags=None):
        with torch.set_grad_enabled(self.training):
            return self._forward(blob_conv, rpn_ret, adv_target, flags)

    def _forward(self, blob_conv, rpn_ret, adv_target=None, flags=None):
        return_dict = {}

        batch_size = blob_conv.size(0)
        if self.training and cfg.DEBUG:
            # debug: batch_size and fg-fraction
            fg = len([x for x in rpn_ret['labels_int32'] if x > 0])
            print("\tbatch-size in discriminator: {} (fg: {}%)".format(batch_size,
                                                                     1.0 * fg / batch_size * 100.0))

            print("\tBlob_conv size in discriminator: {}".format(blob_conv.view(batch_size, -1).size()))

        blob_conv_flattened = blob_conv.view(batch_size, -1)

        adv_score = self.adversarial(blob_conv_flattened)

        box_feat = self.Box_Head(blob_conv_flattened)
        cls_score, bbox_pred = self.Box_Outs(box_feat)

        if self.training:
            if adv_target is None:
                raise ValueError("adv_target must not be None during training!!")

            return_dict['losses'] = {}
            return_dict['metrics'] = {}

            if cfg.GAN.TRAIN.IGNORE_BG_ADV_LOSS:
                # ignore adversarial loss for background RoIs
                mask = np.where(rpn_ret['labels_int32'] == 0)
                fg = len([x for x in rpn_ret['labels_int32'] if x > 0])
                bg = len([x for x in rpn_ret['labels_int32'] if x == 0])
                if cfg.DEBUG:
                    print("ignoring backgound rois in adv_loss: {} / {}".format(bg,
                                                                                len(rpn_ret['labels_int32'])))

                loss_adv = self.adversarial_loss(adv_score, adv_target,
                                                 reduce=False)
                loss_adv[mask] = 0.0
                if fg > 0:
                    loss_adv = loss_adv * len(rpn_ret['labels_int32']) / fg
                loss_adv = loss_adv.mean()
            else:
                loss_adv = self.adversarial_loss(adv_score, adv_target)

            return_dict['losses']['loss_adv'] = loss_adv

            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:  # if testing
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
            return_dict['adv_score'] = adv_score
            return_dict['rois'] = rpn_ret['rois']
            return_dict['rpn_ret'] = rpn_ret

        return return_dict

    def _init_modules(self, pretrained_weights=None):
        """
        inits layers and loads pretrained detectron-backbone-architecture (Conv_Body and RPN)
        also freezes weights of Conv_Body and RPN
        """
        if pretrained_weights is not None:

            pretrained_detectron = torch.load(pretrained_weights)
            load_layers = ['Box_Head', 'Box_Outs']
            mapping, _ = self.detectron_weight_mapping()
            state_dict = {}
            ckpt = pretrained_detectron['model']
            for name in ckpt:
                if name.split('.')[0] in load_layers:
                    if "fc_head" in name:
                        try:
                            if mapping[name]:
                                state_dict[name] = ckpt[name]
                        except KeyError:
                            name_parts = name.split('.')
                            name_parts = [x for x in name_parts if x != "fc_head"]
                            name_modified = '.'.join(name_parts)
                            if mapping[name_modified]:
                                state_dict[name_modified] = ckpt[name]
                    else:
                        try:
                            if mapping[name]:
                                state_dict[name] = ckpt[name]
                        except KeyError:
                            name_parts = name.split('.')
                            name_parts.insert(1, "fc_head")
                            name_modified = '.'.join(name_parts)
                            if mapping[name_modified]:
                                state_dict[name_modified] = ckpt[name]
            self.load_state_dict(state_dict, strict=False)
            del pretrained_detectron
            torch.cuda.empty_cache()

    def _init_weights(self):
        """
        initialize layers before ReLU activation with kaiming initialization
        """
        if cfg.GAN.MODEL.KAIMING_INIT:
            if cfg.DEBUG:
                print("\tInit Adversarial with KAIMING")
            init.kaiming_uniform_(self.adversarial[0].weight, a=0, mode='fan_in', nonlinearity='relu')
            init.constant_(self.adversarial[0].bias, 0.0)
            init.kaiming_uniform_(self.adversarial[2].weight, a=0, mode='fan_in', nonlinearity='relu')
            init.constant_(self.adversarial[2].bias, 0.0)
            init.kaiming_uniform_(self.adversarial[4].weight, a=0, mode='fan_in', nonlinearity='relu')
            init.constant_(self.adversarial[4].bias, 0.0)
        else:
            if cfg.DEBUG:
                print("\tInit ResidualBlock with XAVIER")
            mynn.init.XavierFill(self.adversarial[0].weight)
            init.constant_(self.adversarial[0].bias, 0.0)
            mynn.init.XavierFill(self.adversarial[2].weight)
            init.constant_(self.adversarial[2].bias, 0.0)
            mynn.init.XavierFill(self.adversarial[4].weight)
            init.constant_(self.adversarial[4].bias, 0.0)

    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            d_wmap['adversarial.0.weight'] = 'advFc1_w'
            d_wmap['adversarial.0.bias'] = 'advFc1_b'
            d_wmap['adversarial.2.weight'] = 'advFc2_w'
            d_wmap['adversarial.2.bias'] = 'advFc2_b'
            d_wmap['adversarial.4.weight'] = 'advFc3_w'
            d_wmap['adversarial.4.bias'] = 'advFc3_b'

            for name, m_child in self.named_children():
                if name == 'adversarial':
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

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value

    def adversarial_loss(self, blob, target, reduce=True):
        return F.binary_cross_entropy(blob, target, reduce=reduce)
