# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Tuple

import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjectorBase
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class GroupwiseWeightFloatQuantProxyFromInjector(WeightFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseFloat
        self.quant_tensor_class = GroupwiseFloatQuantTensor
        self._group_dim = self.quant_injector.group_dim
        self._group_size = self.quant_injector.group_size
        if self._group_dim is None or self._group_size is None:
            raise RuntimeError(
                "Groupwise quantization requires a resolved group_dim and group_size")

    def _scale(self):
        return self.retrieve_attribute('_scale')

    def _zero_point(self):
        return self.retrieve_attribute('_zero_point')

    @property
    def group_dim(self):
        return self._group_dim

    @property
    def group_size(self):
        return self._group_size

    def apply_input_view(self, x):
        x = super().apply_input_view(x)
        start_dim = self.group_dim if self.group_dim >= 0 else self.group_dim - 1
        return x.flatten(start_dim, start_dim + 1)

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> GroupwiseFloatQuantTensor:
        shape = self.tracked_parameter_list[0].shape
        out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
        return self.quant_tensor_class(
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
