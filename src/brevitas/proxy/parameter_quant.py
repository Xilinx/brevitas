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

from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List
from typing_extensions import Protocol, runtime_checkable

import torch
from torch import Tensor

from brevitas.inject import BaseInjector as Injector
from brevitas.function import max_int
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector, QuantProxyProtocol


__all__ = ['WeightQuantProxyFromInjector',
           'BiasQuantProxyFromInjector']


@runtime_checkable
class ParameterQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def add_tracked_parameter(self, parameter: torch.nn.Parameter) -> None:
        ...


@runtime_checkable
class WeightQuantProxyProtocol(ParameterQuantProxyProtocol, Protocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        ...


@runtime_checkable
class BiasQuantProxyProtocol(ParameterQuantProxyProtocol, Protocol):
    requires_input_bit_width: bool
    requires_input_scale: bool

    def forward(
            self,
            x: Tensor,
            input_scale: Optional[Tensor],
            input_bit_width: Optional[Tensor]) -> QuantTensor:
        ...


class ParameterQuantProxyFromInjector(QuantProxyFromInjector, ParameterQuantProxyProtocol):
    __metaclass__ = ABCMeta

    def __init__(self, quant_injector: Injector) -> None:
        super(ParameterQuantProxyFromInjector, self).__init__(quant_injector)
        self.tensor_quant = None
        if 'tracked_parameter_list' in quant_injector:
            self.tracked_parameter_list = quant_injector.tracked_parameter_list
        else:
            self.tracked_parameter_list = None
        self.init_tensor_quant()

    @property
    def is_quant_enabled(self):
        return self.tensor_quant is not None

    def init_tensor_quant(self):
        if self.tracked_parameter_list is not None and self.tracked_parameter_list:
            self.quant_injector = self.quant_injector.let(
                tracked_parameter_list=self.tracked_parameter_list)
            self.tensor_quant = self.quant_injector.tensor_quant

    def max_uint_value(self, bit_width):
        return max_int(True, self.is_narrow_range, bit_width)

    def add_tracked_parameter(self, parameter: torch.nn.Parameter) -> None:
        if self.tracked_parameter_list is None:
            self.tracked_parameter_list = []
        if parameter is not None:
            self.tracked_parameter_list.append(parameter)
        if self.tensor_quant is not None:
            del self.tensor_quant
        self.init_tensor_quant()

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(ParameterQuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.init_tensor_quant()


class WeightQuantProxyFromInjector(ParameterQuantProxyFromInjector, WeightQuantProxyProtocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_quant_enabled:
            out, scale, zero_point, bit_width = self.tensor_quant(x)
            return QuantTensor(out, scale, zero_point, bit_width, signed=self.is_signed)
        else:  # quantization disabled
            return QuantTensor(x)


class BiasQuantProxyFromInjector(ParameterQuantProxyFromInjector, BiasQuantProxyProtocol):

    @property
    def requires_input_bit_width(self) -> bool:
        return self.quant_injector.requires_input_bit_width

    @property
    def requires_input_scale(self):
        return self.quant_injector.requires_input_scale

    def forward(
            self,
            x: Tensor,
            input_scale: Optional[Tensor] = None,
            input_bit_width: Optional[Tensor] = None) -> QuantTensor:
        if self.is_quant_enabled:
            if self.requires_input_scale and input_scale is None:
                raise RuntimeError("Input scale required")
            if self.requires_input_bit_width and input_bit_width is None:
                raise RuntimeError("Input bit width required")
            if self.requires_input_scale and self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_bit_width, out_zp = self.tensor_quant(x, input_scale, input_bit_width)
            elif self.requires_input_scale and not self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_bit_width, out_zp = self.tensor_quant(x, input_scale)
            elif not self.requires_input_scale and not self.requires_input_bit_width:
                out, out_scale, out_bit_width, out_zp = self.tensor_quant(x)
            else:
                raise RuntimeError("Internally defined bit width required")
            return QuantTensor(out, out_scale, out_bit_width, out_zp, self.is_signed)
        else:
            return QuantTensor(x)

