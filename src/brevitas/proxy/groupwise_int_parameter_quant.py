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
        return self.retrieve_attribute('scale_')

    def zero_point_(self):
        return self.retrieve_attribute('zero_point_')

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def apply_input_view(self, x):
        x = super().apply_input_view(x)
        start_dim = self.group_dim if self.group_dim >= 0 else self.group_dim - 1
        return x.flatten(start_dim, start_dim + 1)

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> GroupwiseIntQuantTensor:
        shape = self.tracked_parameter_list[0].shape
        out, scale, zero_point, bit_width = qt_args
        return GroupwiseIntQuantTensor(
            out,
            scale,
            zero_point,
            self.group_size,
            self.group_dim,
            bit_width,
            self.is_signed,
            self.training,
            shape)
