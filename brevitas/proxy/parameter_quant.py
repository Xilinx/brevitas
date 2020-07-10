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

from abc import ABCMeta
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from dependencies import Injector

from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxy


__all__ = ['WeightQuantProxy', 'BiasQuantProxy']


class ParameterQuantProxy(QuantProxy):
    __metaclass__ = ABCMeta

    @property
    def tensor_quant(self):
        return self._tensor_quant

    @tensor_quant.setter
    def tensor_quant(self, tensor_quant):
        self._tensor_quant = tensor_quant

    @tensor_quant.deleter
    def tensor_quant(self):
        del self._tensor_quant


class WeightQuantProxy(ParameterQuantProxy):

    def __init__(
            self,
            weight_quant_injector: Injector) -> None:
        super(WeightQuantProxy, self).__init__()
        self.weight_quant_injector = weight_quant_injector
        if 'tracked_parameter_list' in weight_quant_injector:
            self.tracked_parameter_list = weight_quant_injector.tracked_parameter_list
        else:
            self.tracked_parameter_list = None
        self.init_tensor_quant()

    def init_tensor_quant(self):
        if self.tracked_parameter_list is not None:
            self.weight_quant_injector = self.weight_quant_injector.let(
                tracked_parameter_list=self.tracked_parameter_list)
        self.tensor_quant = self.weight_quant_injector.tensor_quant

    def add_tracked_parameter(self, weight: torch.nn.Parameter) -> None:
        if self.tracked_parameter_list is not None:
            self._tracked_parameter_list.append(weight)
        else:
            self.tracked_parameter_list = [weight]
        del self.tensor_quant
        self.init_tensor_quant()

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.tensor_quant is not None:
            out, scale, bit_width = self.tensor_quant(x)
            return QuantTensor(out, scale, bit_width, signed=self.weight_quant_injector.signed)
        else:
            return QuantTensor(x, None, None, None)

    def int_weight(self, x: torch.Tensor):
        quant_weight, scale, _ = self.tensor_quant(x)
        quant_weight = quant_weight / scale
        quant_weight = round_ste(quant_weight)
        quant_weight = quant_weight.int()
        return quant_weight

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(WeightQuantProxy, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.init_tensor_quant()


class BiasQuantProxy(ParameterQuantProxy):

    def __init__(self, bias_quant_injector) -> None:
        super(BiasQuantProxy, self).__init__()
        self.tensor_quant = bias_quant_injector.tensor_quant
        self.bias_quant_injector = bias_quant_injector

    def forward(
            self,
            x: Tensor,
            input_scale: Tensor,
            input_bit_width: Optional[Tensor]) -> QuantTensor:
        if self.tensor_quant is not None:
            if input_scale is None:
                raise RuntimeError("Input scale can't be None when quantizing bias")
            input_scale = input_scale.view(-1)
            if self.bias_quant_injector.requires_input_bit_width:  # bit width is defined outside
                if input_bit_width is None:
                    raise RuntimeError("Input or predefined bit width required")
                out, out_scale, out_bit_width = self.tensor_quant(x, input_scale, input_bit_width)
            else:
                out, out_scale, out_bit_width = self.tensor_quant(x, input_scale)
            return QuantTensor(out, out_scale, out_bit_width, self.bias_quant_injector.signed)
        else:
            return QuantTensor(x, None, None, None)


