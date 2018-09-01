import torch.nn as nn
import torch

from modeling.generator import Generator
from modeling.discriminator import Discriminator


class GAN(nn.Module):
    def __init__(self, generator_weights=None, discriminator_weights=None):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.generator = Generator()
        resolution = self.generator.Conv_Body.resolution
        dim_in = self.generator.RPN.dim_out
        self.discriminator = Discriminator(dim_in, resolution)
        self._init_module(generator_weights, discriminator_weights)

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):

        gen_out = self.generator(data, im_info, roidb, **rpn_kwargs)

        blob_conv = gen_out['blob_fake']
        rpn_ret = gen_out['rpn_ret']

        dis_out = self.discriminator(blob_conv, rpn_ret)

        copy_blobs = ['blob_conv_pooled', 'blob_fake', 'bloc_conv_residual']
        for key in copy_blobs:
           dis_out[key] = gen_out[key]

        return dis_out

    def _init_module(self, generator_weights=None, discriminator_weights=None):
        if generator_weights is None or discriminator_weights is None:
            return
        else:
            pretrained_generator = torch.load(generator_weights)
            pretrained_discriminator = torch.load(discriminator_weights)

            ckpt = pretrained_discriminator['model']
            state_dict = {}
            for name in ckpt:
                state_dict[name] = ckpt[name]
            self.discriminator.load_state_dict(state_dict, strict=False)

            ckpt = pretrained_generator['model']
            state_dict = {}
            for name in ckpt:
                state_dict[name] = ckpt[name]
            self.generator.load_state_dict(state_dict, strict=False)

            del pretrained_discriminator
            del pretrained_generator
            torch.cuda.empty_cache()

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