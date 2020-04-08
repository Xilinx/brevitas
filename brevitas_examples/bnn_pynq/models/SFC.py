# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import reduce
from operator import mul

from torch.nn import Module, ModuleList, BatchNorm1d, Dropout
import torch

from .common import get_quant_linear, get_act_quant, get_quant_type, QuantLinear

FC_OUT_FEATURES = [256, 256, 256]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2


class SFC(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width=None,
                 in_bit_width=None, in_ch=1, in_features=(28, 28)):
        super(SFC, self).__init__()

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)

        self.features = ModuleList()
        self.features.append(get_act_quant(in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(get_quant_linear(in_features=in_features,
                                                  out_features=out_features,
                                                  per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                  bit_width=weight_bit_width,
                                                  quant_type=weight_quant_type))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(get_act_quant(act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.features.append(get_quant_linear(in_features=in_features,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=weight_bit_width,
                                   quant_type=weight_quant_type))
        self.features.append(BatchNorm1d(num_features=num_classes))

        for m in self.modules():
          if isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x


def sfc(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    net = SFC(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net