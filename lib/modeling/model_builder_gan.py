import torch.nn as nn
from torch.autograd import Variable
import torch


from modeling.generator import Generator
from modeling.discriminator import Discriminator
from core.config import cfg


class GAN(nn.Module):
    def __init__(self, generator_weights=None, discriminator_weights=None):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.generator = Generator(generator_weights)
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

        if cfg.DEBUG:
            print("forward in model_builder_gan")
            print("data: {}".format(type(data)))
            print("im_info: {}".format(im_info))
            print("roidb: {}".format(type(roidb)))
            print("flags: {}".format(type(flags)))
            print("flags: {}".format(type(adv_target)))
            for key, value in rpn_kwargs:
                print("{}: {}".format(key, type(value)))

        gen_out = self.generator(data, im_info, roidb, flags, **rpn_kwargs)

        if self.training:
            blob_conv = None

            outputs_gen = self.generator(data, im_info, roidb, flags, **rpn_kwargs)

            if flags.real_mode:
                blob_conv = outputs_gen['blob_conv_pooled']
            elif flags.fake_mode:
                blob_conv = outputs_gen['blob_fake']

            rpn_ret = outputs_gen['rpn_ret']

            input_discriminator = {'blob_conv': blob_conv,
                                   'rpn_ret': rpn_ret,
                                   'adv_target': adv_target
                                   }
        else:
            blob_fake = gen_out['blob_fake']
            rpn_ret = gen_out['rpn_ret']
            input_discriminator = {'blob_conv': blob_fake,
                                   'rpn_ret': rpn_ret
                                   }

        dis_out = self.discriminator(**input_discriminator)

        if not self.training: # if eval only
            copy_blobs = ['blob_conv_pooled', 'blob_fake', 'blob_conv_residual', 'rpn_ret']
            for key in copy_blobs:
                dis_out[key] = gen_out[key]

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