# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Tuple

import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.groupwise_proxy_mixin import GroupwiseProxyMixin
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseWeightQuantProxyFromInjector(GroupwiseProxyMixin, WeightQuantProxyFromInjector):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseInt

    def _scale(self):
        return self.retrieve_attribute('_scale')

    def _zero_point(self):
        return self.retrieve_attribute('_zero_point')

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
