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

from brevitas.function import max_int
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector, QuantProxyProtocol


__all__ = [
    'WeightQuantProxyFromInjector',
    'BiasQuantProxyFromInjector',
    'WeightQuantProxyProtocol',
    'BiasQuantProxyProtocol'
]


@runtime_checkable
class WeightQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        ...


@runtime_checkable
class BiasQuantProxyProtocol(QuantProxyProtocol, Protocol):
    requires_input_bit_width: bool
    requires_input_scale: bool

    def forward(
            self,
            x: Tensor,
            input_scale: Optional[Tensor],
            input_bit_width: Optional[Tensor]) -> QuantTensor:
        ...


class ParameterQuantProxyFromInjector(QuantProxyFromInjector):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def tracked_parameter_list(self):
        pass

    def init_tensor_quant(self):
        param_list = self.tracked_parameter_list
        # params might not be there yet, e.g. bias before merging
        if param_list:
            self.quant_injector = self.quant_injector.let(tracked_parameter_list=param_list)
            super(ParameterQuantProxyFromInjector, self).init_tensor_quant()

    def max_uint_value(self, bit_width):
        return max_int(False, self.is_narrow_range, bit_width)


class WeightQuantProxyFromInjector(ParameterQuantProxyFromInjector, WeightQuantProxyProtocol):

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    def scale(self):
        scale = self.__call__(self._zero_hw_sentinel()).scale
        return scale

    def zero_point(self):
        zero_point = self.__call__(self._zero_hw_sentinel()).zero_point
        return zero_point

    def bit_width(self):
        scale = self.__call__(self._zero_hw_sentinel()).bit_width
        return scale

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width = impl(x)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)


class DecoupledWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    def pre_scale(self):
        output_tuple = self.tensor_quant(self._zero_hw_sentinel())
        out, pre_scale, pre_zero_point, scale, zero_point, bit_width = output_tuple
        return pre_scale

    def pre_zero_point(self):
        output_tuple = self.tensor_quant(self._zero_hw_sentinel())
        out, pre_scale, pre_zero_point, scale, zero_point, bit_width = output_tuple
        return pre_zero_point

    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, pre_scale, pre_zero_point, scale, zero_point, bit_width = impl(x)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)


class BiasQuantProxyFromInjector(ParameterQuantProxyFromInjector, BiasQuantProxyProtocol):

    @property
    def tracked_parameter_list(self):
        return [m.bias for m in self.tracked_module_list if m.bias is not None]

    @property
    def requires_input_bit_width(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_bit_width
        else:
            return False

    @property
    def requires_input_scale(self) -> bool:
        if self.is_quant_enabled:
            return self.quant_injector.requires_input_scale
        else:
            return False

    def scale(self):
        if self.requires_input_scale:
            return None
        zhs = self._zero_hw_sentinel()
        scale = self.__call__(zhs, zhs, zhs).scale
        return scale

    def zero_point(self):
        zhs = self._zero_hw_sentinel()
        zero_point = self.__call__(zhs, zhs, zhs).zero_point
        return zero_point

    def bit_width(self):
        if self.requires_input_bit_width:
            return None
        zhs = self._zero_hw_sentinel()
        bit_width = self.__call__(zhs, zhs, zhs).bit_width
        return bit_width

    def forward(
            self,
            x: Tensor,
            input_scale: Optional[Tensor] = None,
            input_bit_width: Optional[Tensor] = None) -> QuantTensor:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            if self.requires_input_scale and input_scale is None:
                raise RuntimeError("Input scale required")
            if self.requires_input_bit_width and input_bit_width is None:
                raise RuntimeError("Input bit-width required")
            if self.requires_input_scale and self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale, input_bit_width)
            elif self.requires_input_scale and not self.requires_input_bit_width:
                input_scale = input_scale.view(-1)
                out, out_scale, out_zp, out_bit_width = impl(x, input_scale)
            elif not self.requires_input_scale and not self.requires_input_bit_width:
                out, out_scale, out_zp, out_bit_width = impl(x)
            else:
                raise RuntimeError("Internally defined bit-width required")
            return QuantTensor(out, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        else:
            return QuantTensor(x, training=self.training)

