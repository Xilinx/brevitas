# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import Union, Type, Optional

import torch
from torch.nn import Linear
from torch.nn.functional import linear

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_uint

from brevitas.proxy import WeightQuantProxy, BiasQuantProxy, ActivationQuantProxy
from brevitas.proxy.spec import WeightQuantSpec, BiasQuantSpec, OutputQuantSpec

from .quant_layer import QuantWeightBiasOutputLayer, OVER_BATCH_OVER_CHANNELS_4D_SHAPE

__all__ = ['QuantLinear']


class QuantLinear(QuantWeightBiasOutputLayer, Linear):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            weight_quant: Union[WeightQuantProxy, WeightQuantSpec] = WeightQuantSpec(),
            bias_quant: Union[BiasQuantProxy, BiasQuantSpec] = BiasQuantSpec(),
            output_quant: Union[ActivationQuantProxy, OutputQuantSpec] = OutputQuantSpec(),
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        Linear.__init__(self, in_features, out_features, bias)
        QuantWeightBiasOutputLayer.__init__(
            self,
            weight=self.weight,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.out_features

    @property
    def returned_scale_shape(self):
        return OVER_BATCH_OVER_CHANNELS_4D_SHAPE

    def inner_forward_impl(self, x, quant_weight, quant_bias):
        output = linear(x, quant_weight, quant_bias)
        return output

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_input_val = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_fc_val = self.weight_quant.tensor_quant.int_quant.max_uint(weight_bit_width)
        max_output_val = max_input_val * max_fc_val * self.in_features
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width



