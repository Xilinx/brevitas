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

from typing import Optional, Union
from typing_extensions import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.nn import Identity, Module

import brevitas
from brevitas.inject import BaseInjector as Injector
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector, QuantProxyProtocol


@runtime_checkable
class ActQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: Union[Tensor, QuantTensor]) -> QuantTensor:
        ...


@runtime_checkable
class AccQuantProxyProtocol(QuantProxyProtocol, Protocol):

    def forward(self, x: QuantTensor) -> QuantTensor:
        ...


class FusedActivationQuantProxy(brevitas.jit.ScriptModule):

    def __init__(self, activation_impl, tensor_quant):
        super(FusedActivationQuantProxy, self).__init__()
        self.activation_impl = activation_impl
        self.tensor_quant = tensor_quant

    @brevitas.jit.script_method
    def forward(self, x):
        x = self.activation_impl(x)
        x, output_scale, output_zp, output_bit_width = self.tensor_quant(x)
        return x, output_scale, output_zp, output_bit_width


class ActQuantProxyFromInjector(QuantProxyFromInjector, ActQuantProxyProtocol):

    def __init__(self, act_quant_injector: Injector):
        super(ActQuantProxyFromInjector, self).__init__(act_quant_injector)
        tensor_quant = act_quant_injector.tensor_quant
        act_impl = act_quant_injector.act_impl
        self.is_quant_enabled = tensor_quant is not None
        self.is_act_enabled = act_impl is not None
        self.passthrough_act = act_quant_injector.passthrough_act
        if self.is_act_enabled and self.is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                act_impl, tensor_quant)
        elif self.is_act_enabled and not self.is_quant_enabled:
            self.fused_activation_quant_proxy = act_impl
        elif not self.is_act_enabled and self.is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                Identity(), tensor_quant)
        else:
            self.fused_activation_quant_proxy = None
        if 'update_state_dict_impl' in act_quant_injector:
            self.update_state_dict_impl = act_quant_injector.update_state_dict_impl
        else:
            self.update_state_dict_impl = None

    def scale(self, force_eval=True):
        current_status = self.training
        if force_eval:
            self.eval()
        scale = self.__call__(self._zero_hw_sentinel()).scale
        self.train(current_status)
        return scale

    def zero_point(self, force_eval=True):
        current_status = self.training
        if force_eval:
            self.eval()
        zero_point = self.__call__(self._zero_hw_sentinel()).zero_point
        self.train(current_status)
        return zero_point

    def bit_width(self):
        scale = self.__call__(self._zero_hw_sentinel()).bit_width
        return scale

    def forward(self, x: Union[Tensor, QuantTensor]) -> QuantTensor:
        if self.is_act_enabled or self.is_quant_enabled:
            y = x
            if isinstance(y, QuantTensor):
                y = y.value
            y = self.fused_activation_quant_proxy(y)
            if isinstance(y, tuple):
                return QuantTensor(*y, signed=self.is_signed)
            elif self.passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                return QuantTensor(y, x.scale, x.zero_point, x.bit_width, x.signed)
            else:
                return QuantTensor(y)
        else:
            if isinstance(x, QuantTensor):  # passthrough
                return x
            else:
                return QuantTensor(x)

    def identity_quant(self):
        return IdentityQuantProxyFromInjector(self.quant_injector.let(act_impl=None))

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.update_state_dict_impl is not None:
            self.update_state_dict_impl(prefix, state_dict)
        super(ActQuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class IdentityQuantProxyFromInjector(ActQuantProxyFromInjector, ActQuantProxyProtocol):

    def __init__(
            self,
            act_quant_injector: Injector):
        assert act_quant_injector.act_impl is None
        super(IdentityQuantProxyFromInjector, self).__init__(act_quant_injector)

    def identity_quant_proxy(self):
        return self


class ClampQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self, clamp_quant_injector: Injector):
        super(ClampQuantProxyFromInjector, self).__init__(clamp_quant_injector)
        tensor_quant = clamp_quant_injector.tensor_quant
        self.is_quant_enabled = tensor_quant is not None
        self.tensor_quant = tensor_quant

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            out_tuple = self.tensor_quant(x.value, x.scale, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return QuantTensor(out_value, out_scale, out_zp, out_bit_width, self.is_signed)
        return x


class TruncQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self, trunc_quant_injector: Injector):
        super(TruncQuantProxyFromInjector, self).__init__(trunc_quant_injector)
        tensor_quant = trunc_quant_injector.tensor_quant
        self.is_quant_enabled = tensor_quant is not None
        self.tensor_quant = tensor_quant

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            out_tuple = self.tensor_quant(x.value, x.scale, x.zero_point, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return QuantTensor(out_value, out_scale, out_zp, out_bit_width, x.signed)
        else:
            return x