# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Tuple

import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjectorBase
from brevitas.proxy.groupwise_proxy_mixin import GroupwiseProxyMixin
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class GroupwiseWeightFloatQuantProxyFromInjector(GroupwiseProxyMixin,
                                                 WeightFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseFloat

    def _scale(self):
        return self.retrieve_attribute('_scale')

    def _zero_point(self):
        return self.retrieve_attribute('_zero_point')

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> GroupwiseFloatQuantTensor:
        shape = self.tracked_parameter_list[0].shape
        out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
        return GroupwiseFloatQuantTensor(
            out,
            scale,
            zero_point,
            self.group_size,
            self.group_dim,
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            self.is_signed,
            self.training,
            shape)
