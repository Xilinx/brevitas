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

import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from .tensor_norm import TensorNorm
from .common import get_quant_conv2d, get_quant_linear, get_act_quant, get_quant_type
from brevitas.nn import QuantConv2d, QuantHardTanh, QuantLinear

from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


# QuantConv2d configuration
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]

# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = False
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]

# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2


class CNV(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width=None, in_bit_width=None, in_ch=3):
        super(CNV, self).__init__()

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        max_in_val = 1-2**(-7) # for Q1.7 input format
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantHardTanh(bit_width=in_bit_width,
                                                quant_type=in_quant_type,
                                                max_val=max_in_val,
                                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                                scaling_impl_type=ScalingImplType.CONST))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(get_quant_conv2d(in_ch=in_ch,
                                                       out_ch=out_ch,
                                                       bit_width=weight_bit_width,
                                                       quant_type=weight_quant_type))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(get_act_quant(act_bit_width, act_quant_type))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(get_quant_linear(in_features=in_features,
                                                         out_features=out_features,
                                                         per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                         bit_width=weight_bit_width,
                                                         quant_type=weight_quant_type))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(get_act_quant(act_bit_width, act_quant_type))
        
        self.linear_features.append(get_quant_linear(in_features=LAST_FC_IN_FEATURES,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=weight_bit_width,
                                   quant_type=weight_quant_type))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    net = CNV(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net

