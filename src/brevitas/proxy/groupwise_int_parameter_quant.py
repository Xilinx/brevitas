from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def create_quant_tensor(self, qt_args: List[Any]) -> Union[Tensor, GroupwiseIntQuantTensor]:
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
