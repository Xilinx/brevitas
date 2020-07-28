from abc import ABCMeta
from typing_extensions import Protocol, runtime_checkable

from torch import tensor, nn
from dependencies import Injector

from brevitas.core.utils import StatelessBuffer


class QuantProxyProtocol(Protocol):
    is_quant_enabled: bool
    is_signed: bool
    is_narrow_range: bool


class QuantProxy(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(QuantProxy, self).__init__()
        self._zero_hw_sentinel = StatelessBuffer(tensor(0.0))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(QuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        # for retrocompatibility with when it wasn't removed
        zero_hw_sentinel_key = prefix + 'zero_hw_sentinel'
        if zero_hw_sentinel_key in unexpected_keys:
            unexpected_keys.remove(zero_hw_sentinel_key)


class QuantProxyFromInjector(QuantProxy):
    __metaclass__ = ABCMeta

    def __init__(self, quant_injector: Injector) -> None:
        super(QuantProxyFromInjector, self).__init__()
        self.quant_injector = quant_injector

    @property
    def is_signed(self):
        return self.quant_injector.signed

    @property
    def is_narrow_range(self):
        return self.quant_injector.narrow_range


