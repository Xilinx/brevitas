from abc import ABCMeta, abstractmethod
from typing import Optional
from typing_extensions import Protocol, runtime_checkable

from torch import tensor, nn

from brevitas.inject import BaseInjector as Injector
from brevitas.core.utils import StatelessBuffer


__all__ = [
    'QuantProxyProtocol',
    'QuantProxyFromInjector',
]


def _is_signed(quant_injector):
    if 'signed' in quant_injector:
        return quant_injector.signed
    return None


def _is_narrow_range(quant_injector):
    if 'narrow_range' in quant_injector:
        return quant_injector.narrow_range
    return None


def _update_state_dict_impl(quant_injector):
    if 'update_state_dict_impl' in quant_injector:
        return quant_injector.update_state_dict_impl
    return None


@runtime_checkable
class QuantProxyProtocol(Protocol):
    is_quant_enabled: bool
    is_signed: Optional[bool]
    is_narrow_range: Optional[bool]

    def add_tracked_module(self, module: nn.Module) -> None:
        ...


class QuantProxyFromInjector(nn.Module, QuantProxyProtocol):
    __metaclass__ = ABCMeta

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super(QuantProxyFromInjector, self).__init__()
        self.is_signed = _is_signed(quant_injector)
        self.is_narrow_range = _is_narrow_range(quant_injector)
        self.update_state_dict_impl = _update_state_dict_impl(quant_injector)
        self.quant_injector = quant_injector
        self._zero_hw_sentinel = StatelessBuffer(tensor(0.0))
        self.tensor_quant = None
        # Use a normal list and not a ModuleList since this is a pointer to parent modules
        self.tracked_module_list = []
        self.add_tracked_module(quant_layer)

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
        return self.tensor_quant is not None

    def add_tracked_module(self, module: nn.Module) -> None:
        if module is not None:
            self.tracked_module_list.append(module)
            self.update_tracked_modules()
            self.init_tensor_quant()
        else:
            raise RuntimeError("Trying to add None as a parent module.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.update_state_dict_impl is not None:
            self.update_state_dict_impl(prefix, state_dict)
        super(QuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # reload tensor_quant on changes of the state_dict
        # this is called after the parent module state_dict is restored (e.g. weights)
        # so init_tensor_quant takes into account new data from the parent module,
        # but before the state_dict of tensor_quant is loaded, so in case e.g. there is a value
        # for the parameter already, it's not overwritten
        self.init_tensor_quant()
        # for retrocompatibility with when it wasn't removed
        zero_hw_sentinel_key = prefix + 'zero_hw_sentinel'
        if zero_hw_sentinel_key in unexpected_keys:
            unexpected_keys.remove(zero_hw_sentinel_key)

