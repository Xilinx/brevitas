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

import ast
from functools import reduce
from operator import mul

from torch.nn import Module, ModuleList, BatchNorm1d, Dropout
import torch

from brevitas.nn import QuantIdentity, QuantLinear
from .common import CommonWeightQuant, CommonActQuant
from .tensor_norm import TensorNorm

DROPOUT = 0.2


class FC(Module):

    def __init__(
            self,
            num_classes,
            weight_bit_width,
            act_bit_width,
            in_bit_width,
            in_channels,
            out_features,
            in_features=(28, 28)):
        super(FC, self).__init__()

        self.features = ModuleList()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width))
        self.features.append(Dropout(p=DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in out_features:
            self.features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant))
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            self.features.append(Dropout(p=DROPOUT))
        self.features.append(QuantLinear(
                in_features=in_features,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant))
        self.features.append(TensorNorm())

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


def fc(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    out_features = ast.literal_eval(cfg.get('MODEL', 'OUT_FEATURES'))
    net = FC(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        in_channels=in_channels,
        out_features=out_features,
        num_classes=num_classes)
    return net