"""Helper functions for loading pretrained weights from Detectron pickle files
"""

import pickle
import re
import torch
import os

import nn as mynn
from core.config import cfg


def load_caffe2_detectron_weights(net, detectron_weight_file):
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    print(name_mapping)
    for p_name, p_tensor in params.items():
        print(p_name)
        d_name = name_mapping[p_name]
        if isinstance(d_name, str):  # maybe str, None or True
            p_tensor.copy_(torch.Tensor(src_blobs[d_name]))


def resnet_weights_name_pattern():
    pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
    return pattern


def vgg16_weights_name_pattern():
    pattern = re.compile(r"conv\d+_\d+_.+")
    return pattern


def vgg_cnn_m_1024_weights_name_pattern():
    pattern = re.compile(r"conv\d+.+")
    return pattern


def convert_resnet_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    for k, v in src_dict.items():
        toks = k.split('.')
        if k.startswith('layer'):
            assert len(toks[0]) == 6
            res_id = int(toks[0][5]) + 1
            name = '.'.join(['res%d' % res_id] + toks[1:])
            dst_dict[name] = v
        elif k.startswith('fc'):
            continue
        else:
            name = '.'.join(['res1'] + toks)
            dst_dict[name] = v
    return dst_dict


def load_caffe2_pretrained_weights(model, pretrained_weight_file):
    """Load pretrained weights
    """
    _, ext = os.path.splitext(pretrained_weight_file)
    if ext == '.pkl':
        with open(pretrained_weight_file, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrained_state_dict = src_blobs
    elif cfg.BACKBONE_TYPE == 'ResNet':
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
        pretrained_state_dict = convert_resnet_state_dict(torch.load(weights_file))

        # Convert batchnorm weights
        for name, mod in model.named_modules():
            if isinstance(mod, mynn.AffineChannel2d):
                if cfg.FPN.FPN_ON:
                    pretrianed_name = name.split('.', 2)[-1]
                else:
                    pretrianed_name = name.split('.', 1)[-1]
                bn_mean = pretrained_state_dict[pretrianed_name + '.running_mean']
                bn_var = pretrained_state_dict[pretrianed_name + '.running_var']
                scale = pretrained_state_dict[pretrianed_name + '.weight']
                bias = pretrained_state_dict[pretrianed_name + '.bias']
                std = torch.sqrt(bn_var + 1e-5)
                new_scale = scale / std
                new_bias = bias - bn_mean * scale / std
                pretrained_state_dict[pretrianed_name + '.weight'] = new_scale

                pretrained_state_dict[pretrianed_name + '.bias'] = new_bias
    else:
        raise ValueError('Invalid weight file specified!')

    model_state_dict = model.state_dict()

    if cfg.TRAIN.BACKBONE_TYPE == 'ResNet':
        pattern = resnet_weights_name_pattern()
    elif cfg.TRAIN.BACKBONE_TYPE == 'VGG16':
        pattern = vgg16_weights_name_pattern()
    elif cfg.TRAIN.BACKBONE_TYPE == 'CNN_M_1024':
        pattern = vgg_cnn_m_1024_weights_name_pattern()
    else:
        raise ValueError("Invalid Backbone Architecture specified!")

    name_mapping, _ = model.detectron_weight_mapping

    for k, v in name_mapping.items():
        if isinstance(v, str):  # maybe a str, None or True
            if pattern.match(v):
                if cfg.FPN.FPN_ON:
                    pretrained_key = k.split('.', 2)[-1]
                else:
                    pretrained_key = k.split('.', 1)[-1]

                if ext == '.pkl':
                    model_state_dict[k].copy_(torch.Tensor(pretrained_state_dict[v]))
                elif cfg.BACKBONE_TYPE == 'ResNet':
                    model_state_dict[k].copy_(pretrained_state_dict[pretrained_key])


if __name__ == '__main__':
    """Testing"""
    from pprint import pprint
    import sys
    sys.path.insert(0, '..')
    from lib.modeling.model_builder import Generalized_RCNN
    from lib.core.config import cfg, cfg_from_file

    cfg.MODEL.NUM_CLASSES = 81
    cfg_from_file('../../cfgs/res50_mask.yml')
    net = Generalized_RCNN()

    # pprint(list(net.state_dict().keys()), width=1)

    mapping, orphans = net.detectron_weight_mapping
    state_dict = net.state_dict()

    for k in mapping.keys():
        assert k in state_dict, '%s' % k

    rest = set(state_dict.keys()) - set(mapping.keys())
    assert len(rest) == 0
