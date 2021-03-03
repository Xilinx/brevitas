from abc import ABCMeta, abstractmethod
from typing import Optional
from typing_extensions import Protocol, runtime_checkable

from torch import tensor, nn

from brevitas.inject import BaseInjector as Injector
from brevitas.core.utils import StatelessBuffer


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
        self.quant_injector = quant_injector
        self._zero_hw_sentinel = StatelessBuffer(tensor(0.0))
        # Use a normal list and not a ModuleList since this is a pointer to parent modules
        # and not a traditional submodule relationship
        self.tracked_module_list = []
        self.add_tracked_module(quant_layer)

    def init_tensor_quant(self):
        if len(self.tracked_module_list) == 1:
            self.quant_injector = self.quant_injector.let(module=self.tracked_module_list[0])
        else:
            self.quant_injector = self.quant_injector.let(module=self.tracked_module_list)
        self.tensor_quant = self.quant_injector.tensor_quant
        self.is_quant_enabled = self.tensor_quant is not None

    @property
    def is_signed(self):
        if 'signed' in self.quant_injector:
            return self.quant_injector.signed
        else:
            return None

    @property
    def is_narrow_range(self):
        if 'narrow_range' in self.quant_injector:
            return self.quant_injector.narrow_range
        else:
            return None

    def add_tracked_module(self, module: nn.Module) -> None:
        if module is not None:
            self.tracked_module_list.append(module)
            self.init_tensor_quant()
        else:
            raise RuntimeError("Trying to add None as a parent module.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(QuantProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # reload tensor_quant on changes of the state_dict
        self.init_tensor_quant()
        # for retrocompatibility with when it wasn't removed
        zero_hw_sentinel_key = prefix + 'zero_hw_sentinel'
        if zero_hw_sentinel_key in unexpected_keys:
            unexpected_keys.remove(zero_hw_sentinel_key)

