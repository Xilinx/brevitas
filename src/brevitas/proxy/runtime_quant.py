# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import Optional, Tuple, Union

from torch import nn
from torch import Tensor
from torch.nn import Identity
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

import brevitas
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIO

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
        return self.tensor_quant(x)


class ActQuantProxyFromInjectorBase(QuantProxyFromInjector, ActQuantProxyProtocol, ABC):

    def __init__(self, quant_layer, quant_injector):
        QuantProxyFromInjector.__init__(self, quant_layer, quant_injector)
        ActQuantProxyProtocol.__init__(self)
        self.is_passthrough_act = _is_passthrough_act(quant_injector)
        self._cached_act = None
        self.cache_inference_quant_act = False
        self.cache_quant_io_metadata_only = True

    def internal_forward(self, force_eval):
        current_status = self.training
        if force_eval:
            self.eval()
        out = self.__call__(self._zero_hw_sentinel())
        self.train(current_status)
        return out

    def retrieve_attribute(self, attribute, force_eval):
        if self.is_quant_enabled:
            out = self.internal_forward(force_eval)
            return getattr(out, attribute)
        elif self._cached_act is not None:
            return getattr(self._cached_act, attribute)
        elif self._cached_act is None:
            return None

    @property
    def is_quant_enabled(self):
        return self._is_quant_enabled and not self.disable_quant

    # @property
    # def is_signed(self):
    #     if self._cached_act is not None:
    #         return self._cached_act.signed
    #     try:
    #         return super().is_signed
    #     except:
    #         return None

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


class ActQuantProxyFromInjector(ActQuantProxyFromInjectorBase):

    def scale(self, force_eval=True):
        return self.retrieve_attribute('scale', force_eval)

    def zero_point(self, force_eval=True):
        return self.retrieve_attribute('zero_point', force_eval)

    def bit_width(self, force_eval=True):
        return self.retrieve_attribute('bit_width', force_eval)

    def forward(self, x: Union[Tensor, QuantTensor]) -> Union[Tensor, IntQuantTensor]:
        out = x
        if self.fused_activation_quant_proxy is not None:
            y = x
            if isinstance(y, QuantTensor):
                y = y.value

            if self.export_mode:
                y = self.fused_activation_quant_proxy.activation_impl(y)
                y = self.export_handler(y)
            elif not self.is_quant_enabled:
                y = self.fused_activation_quant_proxy.activation_impl(y)
            else:
                y = self.fused_activation_quant_proxy(y)
            # If y is an empty IntQuantTensor, we need to check if this is a passthrough proxy,
            # otherwise return a simple Tensor
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y)):
                out = IntQuantTensor(
                    *y,
                    signed=self.is_signed,
                    training=self.training,
                    _zero_zero_point=self.is_zero_zero_point)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                if isinstance(y, tuple):
                    y = y[0]
                if isinstance(x, IntQuantTensor):
                    out = IntQuantTensor(
                        y,
                        x.scale,
                        x.zero_point,
                        x.bit_width,
                        x.signed,
                        self.training,
                        x._zero_zero_point)
                else:
                    out = y
            else:
                if isinstance(y, tuple):
                    y = y[0]
                out = y
        else:
            # If fused activation quant proxy is not enabled, return the input
            out = x
        if not self.training and self.cache_inference_quant_act and isinstance(out, IntQuantTensor):
            cached_out = _CachedIO(out.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_out
        return out


class DynamicActQuantProxyFromInjector(ActQuantProxyFromInjector):

    def scale(self, force_eval=True):
        raise RuntimeError("Scale for Dynamic Act Quant is input-dependant")

    def zero_point(self, force_eval=True):
        raise RuntimeError("Zero point for Dynamic Act Quant is input-dependant")


class ClampQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def forward(self, x: IntQuantTensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            out_tuple = self.tensor_quant(x.value, x.scale, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return IntQuantTensor(
                out_value, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        return x


class TruncQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def bit_width(self):
        if not self.is_quant_enabled:
            return None
        zhs = self._zero_hw_sentinel()
        # Signed might or might not be defined. We just care about retrieving the bitwidth
        empty_imp = IntQuantTensor(zhs, zhs, zhs, zhs, signed=True, training=self.training)
        bit_width = self.__call__(empty_imp).bit_width
        return bit_width

    def forward(self, x: IntQuantTensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            if self.export_mode:
                out_tuple = self.export_handler(
                    x.value, x.scale, x.zero_point, x.bit_width, x.signed)
            else:
                out_tuple = self.tensor_quant(x.value, x.scale, x.zero_point, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            return IntQuantTensor(
                out_value, out_scale, out_zp, out_bit_width, x.signed, self.training)
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
