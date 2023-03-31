# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Union

from torch import nn
from torch import Tensor
from torch.nn import Identity
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

import brevitas
from brevitas.quant_tensor import QuantTensor

from .quant_proxy import QuantProxyFromInjector
from .quant_proxy import QuantProxyProtocol

__all__ = [
    'ActQuantProxyProtocol',
    'AccQuantProxyProtocol',
    'ActQuantProxyFromInjector',
    'TruncQuantProxyFromInjector',
    'ClampQuantProxyFromInjector']


def _is_passthrough_act(quant_injector):
    if 'act_impl' not in quant_injector:
        return True
    elif quant_injector.act_impl is None:
        return True
    elif 'passthrough_act' in quant_injector:
        return quant_injector.passthrough_act
    else:
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


class _TensorQuantDisabledIdentity(brevitas.jit.ScriptModule):

    def __init__(self, module_to_wrap=None):
        super(_TensorQuantDisabledIdentity, self).__init__()

    @brevitas.jit.script_method
    def forward(self,
                x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        return (x, None, None, None)


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
        QuantProxyFromInjector.__init__(self, quant_layer, quant_injector)
        ActQuantProxyProtocol.__init__(self)
        self.is_passthrough_act = _is_passthrough_act(quant_injector)

    @property
    def is_quant_enabled(self):
        return self._is_quant_enabled

    @is_quant_enabled.setter
    def is_quant_enabled(self, is_quant_enabled):
        self._is_quant_enabled = is_quant_enabled

    def init_tensor_quant(self):
        tensor_quant = self.quant_injector.tensor_quant
        if 'act_impl' in self.quant_injector:
            act_impl = self.quant_injector.act_impl
        else:
            act_impl = None
        is_act_enabled = _is_act_enabled(act_impl, tensor_quant)
        is_quant_enabled = tensor_quant is not None
        self.is_quant_enabled = is_quant_enabled
        if is_act_enabled and is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(act_impl, tensor_quant)
        elif is_act_enabled and not is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(
                act_impl, _TensorQuantDisabledIdentity())
        elif not is_act_enabled and is_quant_enabled:
            self.fused_activation_quant_proxy = FusedActivationQuantProxy(Identity(), tensor_quant)
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
            # If y is an empty QuantTensor, we need to check if this is a passthrough proxy,
            # otherwise return an empty QuantTensor
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y)):
                return QuantTensor(*y, signed=self.is_signed, training=self.training)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                if isinstance(y, tuple):
                    y = y[0]
                return QuantTensor(y, x.scale, x.zero_point, x.bit_width, x.signed, self.training)
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
            if self.export_mode:
                out_tuple = self.export_handler(
                    x.value, x.scale, x.zero_point, x.bit_width, x.signed)
            else:
                out_tuple = self.tensor_quant(x.value, x.scale, x.zero_point, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return QuantTensor(out_value, out_scale, out_zp, out_bit_width, x.signed, self.training)
        else:
            return x

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
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
