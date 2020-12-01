#
# Based on: https://arxiv.org/abs/1801.04540
#

import torch.nn as nn
import math
import torch

try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_int
from .mixin.base import QuantLayerMixin


class HadamardClassifier(QuantLayerMixin, nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 fixed_scale=False,
                 return_quant_tensor: bool = False):
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)
        nn.Module.__init__(self)
        if hadamard is None:
            raise Exception("Hadamard layer requires scipy to be installed.")

        self.out_channels = out_channels
        self.in_channels = in_channels
        sz = 2 ** int(math.ceil(math.log(max(in_channels, out_channels), 2)))
        mat = torch.from_numpy(hadamard(sz)).float()
        self.register_buffer('proj', mat)
        init_scale = 1. / math.sqrt(self.out_channels)
        if fixed_scale:
            self.register_buffer('scale', torch.tensor(init_scale))
        else:
            self.scale = nn.Parameter(torch.tensor(init_scale))
        self.eps = 1e-8

    def forward(self, x):
        output_scale = None
        output_bit_width = None
        x, input_scale, input_bit_width = self.unpack_input(x)
        norm = x.norm(p='fro', keepdim=True) + self.eps
        x = x / norm
        out = - self.scale * nn.functional.linear(x, self.proj[:self.out_channels, :self.in_channels])
        if self.compute_output_scale:
            output_scale = input_scale * self.scale / norm
        if self.compute_output_bit_width:
            output_bit_width = self.max_output_bit_width(input_bit_width)
        return self.pack_output(out, output_scale, output_bit_width)

    def max_output_bit_width(self, input_bit_width):
        max_input_val = max_int(bit_width=input_bit_width, narrow_range=False, signed=False)
        max_output_val = max_input_val * self.in_channels
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(HadamardClassifier, self).state_dict(destination, prefix, keep_vars)
        del state_dict[prefix + 'proj']
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(HadamardClassifier, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        proj_key = prefix + 'proj'
        if proj_key in missing_keys:
            missing_keys.remove(proj_key)