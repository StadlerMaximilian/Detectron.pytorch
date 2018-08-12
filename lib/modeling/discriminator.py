import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Discriminator(nn.Module):
    def __init__(self, dim_in, pooled_resolution, rpn_dim_out, resolution):
        super().__init__()
        self.fc_dim = dim_in * pooled_resolution * pooled_resolution
        self.adversarial = nn.Sequential(nn.Linear(self.fc_dim, 4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, 1024),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, 1),
                                         nn.Sigmoid())

        self.perceptual_head = get_func(cfg.GAN.MODEL.CONV_BODY_FC_HEAD)(rpn_dim_out, resolution)
        self.perceptual_outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)


    def forward(self, x):
        batch_size = x.size(0)
        y = self.adversarial(x.view(batch_size, -1))

        box_feat = self.Box_Head(blob_conv, rpn_ret)
        cls_score, bbox_pred = self.Box_Outs(box_feat)

        return y