from abc import ABCMeta

from torch import nn

from brevitas.core import ZERO_HW_SENTINEL_NAME


class QuantProxy(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(QuantProxy, self).__init__()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(QuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
            unexpected_keys.remove(zero_hw_sentinel_key)
