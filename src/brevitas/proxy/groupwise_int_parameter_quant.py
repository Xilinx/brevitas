from typing import Any, Tuple

import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseInt

    def scale_(self):
        if not self.is_quant_enabled:
            return None
        elif self._cached_weight:
            scale = self._cached_weight.scale_
        else:
            scale = self.__call__(self.tracked_parameter_list[0]).scale_
        return scale

    def zero_point_(self):
        if not self.is_quant_enabled:
            return None
        elif self._cached_weight:
            zero_point = self._cached_weight.zero_point_
        else:
            zero_point = self.__call__(self.tracked_parameter_list[0]).zero_point_
        return zero_point

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def apply_input_view(self, x):
        x = super().apply_input_view(x)
        start_dim = self.group_dim if self.group_dim != -1 else -2
        return x.flatten(start_dim, start_dim + 1)

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> GroupwiseIntQuantTensor:
        out, scale, zero_point, bit_width = qt_args
        return GroupwiseIntQuantTensor(
            out,
            scale,
            zero_point,
            self.group_size,
            self.group_dim,
            bit_width,
            self.is_signed,
            self.training)
