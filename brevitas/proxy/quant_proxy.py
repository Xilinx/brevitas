from abc import ABCMeta

import torch
from torch import nn

from brevitas.core import ZERO_HW_SENTINEL_NAME, ZERO_HW_SENTINEL_VALUE


class QuantProxy(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(QuantProxy, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(QuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
        if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
            unexpected_keys.remove(zero_hw_sentinel_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(QuantProxy, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + ZERO_HW_SENTINEL_NAME]
        return output_dict
