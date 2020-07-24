# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
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

from typing import Union, Type, Optional

from dependencies import Injector
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_uint
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol, BiasQuantProxyProtocol
from brevitas.proxy.runtime_quant import ActQuantProxyProtocol
from brevitas.quant_tensor import QuantTensor
from .quant_layer import DefaultWeightQuantInjector as DefaultWeightQI
from .quant_layer import DefaultBiasQuantInjector as DefaultBiasQI
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

__all__ = ['ScaleBias', 'QuantScaleBias']


class ScaleBias(Module):

    def __init__(self, num_features: int, bias: bool, runtime_shape=(1, -1, 1, 1)):
        super(ScaleBias, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features)) if bias else None
        self.runtime_shape = runtime_shape

    def forward(self, x):
        return x * self.weight.view(self.runtime_shape) + self.bias.view(self.runtime_shape)


class QuantScaleBias(QuantWBIOL, ScaleBias):

    def __init__(
            self,
            num_features: int,
            bias: bool,
            weight_quant: Union[WeightQuantProxyProtocol, Type[Injector]] = DefaultWeightQI,
            bias_quant: Union[BiasQuantProxyProtocol, Type[Injector]] = DefaultBiasQI,
            input_quant: Union[ActQuantProxyProtocol, Type[Injector]] = None,
            output_quant: Union[ActQuantProxyProtocol, Type[Injector]] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        ScaleBias.__init__(self, num_features, bias)
        QuantWBIOL.__init__(
            self,
            self.weight,
            self.bias,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

    @property
    def per_elem_ops(self):
        return 2

    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.num_features

    @property
    def channelwise_separable(self) -> bool:
        return True

    def quant_weight(self):
        return self.weight_quant(self.weight.view(-1, 1))  # TODO check if the view is needed

    def forward(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(inp)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        quant_weight = quant_weight.view(self.runtime_shape)
        quant_bias = quant_bias.view(self.runtime_shape)
        output_tensor = x * quant_weight + quant_bias
        return output_tensor

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_input_val = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_weight_val = self.weight_quant.max_uint_value(weight_bit_width)
        max_output_val = max_input_val * max_weight_val
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width







