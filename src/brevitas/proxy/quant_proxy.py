# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from torch import nn
from torch import tensor
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas import config
from brevitas.common import ExportMixin
from brevitas.core.utils import StatelessBuffer
from brevitas.inject import BaseInjector as Injector
from brevitas.utils.quant_utils import float_to_int_impl_to_enum

__all__ = [
    'QuantProxyProtocol',
    'QuantProxyFromInjector',]


def _is_signed(quant_injector):
    if 'signed' in quant_injector:
        return quant_injector.signed
    return None


def _is_narrow_range(quant_injector):
    if 'narrow_range' in quant_injector:
        return quant_injector.narrow_range
    return None


def _rounding_mode(quant_injector):
    if 'float_to_int_impl_type' in quant_injector:
        return str(quant_injector.float_to_int_impl_type)
    elif 'float_to_int_impl' in quant_injector:
        try:
            impl_type = float_to_int_impl_to_enum(quant_injector.float_to_int_impl)
            return str(impl_type).upper()
        except:
            return None
    else:
        return None


def _update_state_dict_impl(quant_injector):
    try:
        impl = quant_injector.update_state_dict_impl
    except:
        impl = None
    return impl


@runtime_checkable
class QuantProxyProtocol(Protocol):
    is_quant_enabled: bool
    is_signed: Optional[bool]
    is_narrow_range: Optional[bool]
    rounding_mode: Optional[str]

    def add_tracked_module(self, module: nn.Module) -> None:
        ...


class QuantProxyFromInjector(ExportMixin, nn.Module, QuantProxyProtocol):
    __metaclass__ = ABCMeta

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        ExportMixin.__init__(self)
        nn.Module.__init__(self)
        QuantProxyProtocol.__init__(self)
        self.update_state_dict_impl = _update_state_dict_impl(quant_injector)
        self.quant_injector = quant_injector
        self._zero_hw_sentinel = StatelessBuffer(tensor(0.0))
        self.tensor_quant = None
        # Use a normal list and not a ModuleList since this is a pointer to parent modules
        self.tracked_module_list = []
        self.add_tracked_module(quant_layer)
        self.disable_quant = False

    @property
    def requires_export_handler(self):
        return self.is_quant_enabled

    def update_tracked_modules(self):
        """Update the modules tracked by the injector with the modules tracked by the proxy"""
        if len(self.tracked_module_list) == 1:
            self.quant_injector = self.quant_injector.let(module=self.tracked_module_list[0])
        else:
            # set the list in the injector as a tuple to avoid dealing with inplace modifications
            self.quant_injector = self.quant_injector.let(module=tuple(self.tracked_module_list))

    def init_tensor_quant(self):
        self.tensor_quant = self.quant_injector.tensor_quant

    @property
    def is_quant_enabled(self):
        return not self.disable_quant and self.tensor_quant is not None

    @property
    def is_signed(self):
        return _is_signed(self.quant_injector)

    @property
    def is_narrow_range(self):
        return _is_narrow_range(self.quant_injector)

    @property
    def rounding_mode(self):
        return _rounding_mode(self.quant_injector)

    def add_tracked_module(self, module: nn.Module) -> None:
        if module is not None:
            self.tracked_module_list.append(module)
            self.update_tracked_modules()
            self.init_tensor_quant()
        else:
            raise RuntimeError("Trying to add None as a parent module.")

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        if self.update_state_dict_impl is not None:
            self.update_state_dict_impl(prefix, state_dict)
        super(QuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # reload tensor_quant on changes of the state_dict
        # this is called after the parent module state_dict is restored (e.g. weights)
        # so init_tensor_quant takes into account new data from the parent module,
        # but before the state_dict of tensor_quant is loaded, so in case e.g. there is a value
        # for the parameter already, it's not overwritten
        if config.REINIT_ON_STATE_DICT_LOAD:
            self.init_tensor_quant()
        # for retrocompatibility with when it wasn't removed
        zero_hw_sentinel_key = prefix + 'zero_hw_sentinel'
        if zero_hw_sentinel_key in unexpected_keys:
            unexpected_keys.remove(zero_hw_sentinel_key)
