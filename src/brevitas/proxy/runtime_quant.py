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

from torch import Tensor, nn
from torch.nn import Identity

import brevitas
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector, QuantProxyProtocol



__all__ = [
    'ActQuantProxyProtocol',
    'AccQuantProxyProtocol',
    'ActQuantProxyFromInjector',
    'TruncQuantProxyFromInjector',
    'ClampQuantProxyFromInjector'
]


def _is_passthrough_act(quant_injector):
    if 'passthrough_act' in quant_injector:
        return quant_injector.passthrough_act
    return False


def _is_act_enabled(act_impl, tensor_quant):
    if act_impl is None:
        return False
    # avoid enabling HardTanh when clamping from quantization is already enabled
    elif isinstance(act_impl, nn.Hardtanh) and tensor_quant is not None:
        return False
    else:
        return True


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

    def __init__(self, quant_layer, quant_injector):
        super(ActQuantProxyFromInjector, self).__init__(quant_layer, quant_injector)
        self.is_passthrough_act = _is_passthrough_act(quant_injector)

    @property
    def is_quant_enabled(self):
        return self._is_quant_enabled

    @is_quant_enabled.setter
    def is_quant_enabled(self, is_quant_enabled):
        self._is_quant_enabled = is_quant_enabled

    def init_tensor_quant(self):
        tensor_quant = self.quant_injector.tensor_quant
        act_impl = self.quant_injector.act_impl
        is_act_enabled = _is_act_enabled(act_impl, tensor_quant)
        is_quant_enabled = tensor_quant is not None
        self.is_quant_enabled = is_quant_enabled
        if is_act_enabled and is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                act_impl, tensor_quant)
        elif is_act_enabled and not is_quant_enabled:
            self.fused_activation_quant_proxy = act_impl
        elif not is_act_enabled and is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                Identity(), tensor_quant)
        else:
            self.fused_activation_quant_proxy = None

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
        if self.fused_activation_quant_proxy is not None:
            y = x
            if isinstance(y, QuantTensor):
                y = y.value
            if self.export_mode:
                y = self.fused_activation_quant_proxy.activation_impl(y)
                y = self.export_handler(y)
            else:
                y = self.fused_activation_quant_proxy(y)
            if isinstance(y, tuple):
                return QuantTensor(*y, signed=self.is_signed, training=self.training)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                return QuantTensor(
                    y, x.scale, x.zero_point, x.bit_width, x.signed, self.training)
            else:
                return QuantTensor(y, training=self.training)
        else:
            if isinstance(x, QuantTensor):  # passthrough
                return x
            else:
                return QuantTensor(x, training=self.training)


class ClampQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            out_tuple = self.tensor_quant(x.value, x.scale, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return QuantTensor(
                out_value, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        return x


class TruncQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def bit_width(self):
        zhs = self._zero_hw_sentinel()
        empty_imp = QuantTensor(zhs, zhs, zhs, zhs)
        bit_width = self.__call__(empty_imp).bit_width
        return bit_width

    def forward(self, x: QuantTensor):
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out_tuple = impl(x.value, x.scale, x.zero_point, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return QuantTensor(
                out_value, out_scale, out_zp, out_bit_width, x.signed, self.training)
        else:
            return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(TruncQuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # for retrocompatibility with when it wasn't removed and it was called differently
        zhs = 'zero_hw_sentinel'
        zhs_key = prefix + zhs
        zhs_old_prefix_key = '.'.join(prefix.split('.')[:-2]) + '.accumulator_quant.' + zhs
        if zhs in unexpected_keys:
            unexpected_keys.remove(zhs_key)
        if zhs_old_prefix_key in unexpected_keys:
            unexpected_keys.remove(zhs_old_prefix_key)