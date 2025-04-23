# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import Identity
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

import brevitas
from brevitas import is_dynamo_compiling
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

    def forward(self, x: Union[Tensor, IntQuantTensor]) -> Union[Tensor, IntQuantTensor]:
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
        self.cache_class = None
        self.skip_create_quant_tensor = False

    def compile_quant(self, compile_export=False):
        fullgraph = not self.is_groupwise
        if compile_export and hasattr(self, 'export_handler') and self.export_handler is not None:
            self.export_handler = torch.compile(
                self.export_handler, dynamic=True, fullgraph=fullgraph)
        elif self.fused_activation_quant_proxy is not None:
            self.fused_activation_quant_proxy.tensor_quant = torch.compile(
                self.fused_activation_quant_proxy.tensor_quant, dynamic=True, fullgraph=fullgraph)

    @property
    def input_view_impl(self):
        if self.fused_activation_quant_proxy.tensor_quant is not None and not isinstance(
                self.fused_activation_quant_proxy.tensor_quant, _TensorQuantDisabledIdentity):
            return self.fused_activation_quant_proxy.tensor_quant.int_quant.input_view_impl
        else:
            return Identity()

    def internal_forward(self, force_eval):
        current_status = self.training
        if force_eval:
            self.eval()
        out = self.__call__(self._zero_hw_sentinel())
        self.train(current_status)
        return out

    def retrieve_attribute(self, attribute, force_eval):
        if self._cached_act is not None:
            return getattr(self._cached_act, attribute)
        elif self.is_quant_enabled:
            out = self.internal_forward(force_eval)
            return getattr(out, attribute)
        elif self._cached_act is None:
            return None

    def apply_input_view(self, x):
        return self.input_view_impl(x)

    @property
    def is_quant_enabled(self):
        return self._is_quant_enabled and not self.disable_quant

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

    @abstractmethod
    def create_quant_tensor(
            self, qt_args: Union[torch.Tensor, Tuple[Any]], x: Union[Tensor,
                                                                     QuantTensor]) -> QuantTensor:
        # Supports the following:
        # - qt_args as tuple of Tensors and bools = standard quant activations
        # - qt_args as Tensor and x as QuantTensor = passthrough activation
        # In both cases, the output is a QuantTensor
        raise NotImplementedError

    def forward(self, x: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        # If fused activation quant proxy is not enabled, return the input
        if self.fused_activation_quant_proxy is None:
            return x

        y = x
        if isinstance(y, QuantTensor):
            y = y.value

        if not self.is_quant_enabled:
            # A tuple helps later with control flows
            # The second None value is used later
            y = self.fused_activation_quant_proxy.activation_impl(y)
            y = (y, None)
        elif self.export_mode:
            y = self.fused_activation_quant_proxy.activation_impl(y)
            y = self.export_handler(y)
        else:
            y = self.fused_activation_quant_proxy(y)
        # If y is an empty QuantTensor, we need to check if this is a passthrough proxy,
        # otherwise return a simple Tensor

        if self.skip_create_quant_tensor:
            out = y[0]
        else:
            # If the second value (i.e., scale) is None, then quant is disabled
            if y[1] is not None:
                out = self.create_quant_tensor(y, x=x)
            elif self.is_passthrough_act and isinstance(x, QuantTensor):
                # preserve scale/zp/bit/sign even without output quant
                y = y[0]
                out = self.create_quant_tensor(y, x=x)
            else:
                out = y[0]

        if not self.training and self.cache_inference_quant_act and isinstance(out, QuantTensor):
            cached_out = self.cache_class(out.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_out
        return out


class ActQuantProxyFromInjector(ActQuantProxyFromInjectorBase):

    def __init__(self, quant_layer, quant_injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIO

    def scale(self, force_eval=True):
        return self.retrieve_attribute('scale', force_eval)

    def zero_point(self, force_eval=True):
        return self.retrieve_attribute('zero_point', force_eval)

    def bit_width(self, force_eval=True):
        return self.retrieve_attribute('bit_width', force_eval)

    def create_quant_tensor(
            self, qt_args: Union[torch.Tensor, Tuple[Any]],
            x: Union[Tensor, IntQuantTensor]) -> IntQuantTensor:
        if isinstance(qt_args, tuple):
            out = IntQuantTensor(*qt_args, self.is_signed, self.training)
        else:
            out = IntQuantTensor(
                qt_args, x.scale, x.zero_point, x.bit_width, x.signed, self.training)
        return out


class DynamicActQuantProxyFromInjector(ActQuantProxyFromInjector):

    def scale(self, force_eval=True):
        raise RuntimeError("Scale for Dynamic Act Quant is input-dependant")

    def zero_point(self, force_eval=True):
        raise RuntimeError("Zero point for Dynamic Act Quant is input-dependant")


class ClampQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = False

    def forward(self, x: IntQuantTensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            out_tuple = self.tensor_quant(x.value, x.scale, x.bit_width)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            if self.skip_create_quant_tensor:
                return out_value
            return IntQuantTensor(
                out_value, out_scale, out_zp, out_bit_width, self.is_signed, self.training)
        return x


class TruncQuantProxyFromInjector(QuantProxyFromInjector, AccQuantProxyProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_act = None
        self.cache_inference_quant_act = True
        self.cache_quant_io_metadata_only = True
        self.cache_class = _CachedIO
        self.skip_create_quant_tensor = False

    def retrieve_attribute(self, attribute):
        if self._cached_act is not None:
            return getattr(self._cached_act, attribute)
        elif self._cached_act is None:
            return None

    @property
    def is_narrow_range(self):
        narrow_range = super(TruncQuantProxyFromInjector, self).is_narrow_range
        return narrow_range if narrow_range is not None else False

    def scale(self):
        return self.retrieve_attribute('scale')

    def zero_point(self):
        return self.retrieve_attribute('zero_point')

    def bit_width(self):
        return self.retrieve_attribute('bit_width')

    def forward(self, x: IntQuantTensor) -> Union[Tensor, IntQuantTensor]:
        if self.is_quant_enabled:
            if self.export_mode:
                out_tuple = self.export_handler(
                    x.value, x.scale, x.zero_point, x.bit_width, x.signed)
            else:
                out_tuple = self.tensor_quant(x.value, x.scale, x.zero_point, x.bit_width, x.signed)
            out_value, out_scale, out_zp, out_bit_width = out_tuple
            if self.skip_create_quant_tensor:
                return out_value
            out = IntQuantTensor(
                out_value, out_scale, out_zp, out_bit_width, x.signed, self.training)
            if not self.training and self.cache_inference_quant_act:
                cached_out = self.cache_class(out.detach(), self.cache_quant_io_metadata_only)
                self._cached_act = cached_out
            return out
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
