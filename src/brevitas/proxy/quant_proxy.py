from abc import ABCMeta
from typing_extensions import Protocol, runtime_checkable

from torch import tensor, nn

from brevitas.inject import BaseInjector as Injector
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
        if 'signed' in quant_injector:
            self.is_signed = quant_injector.signed
        else:
            self.is_signed = None
        if 'narrow_range' in quant_injector:
            self.is_narrow_range = self.quant_injector.narrow_range
        else:
            self.is_narrow_range = None

