# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from brevitas.core.function_wrapper.shape import extract_groupwise_block
from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseInt
        self._refresh_region_quant()

    def _refresh_region_quant(self):
        self.supports_quant_weight_region = (
            bool(getattr(self.tensor_quant.scaling_impl, 'supports_groupwise_region', False)) and
            bool(getattr(self.tensor_quant.zero_point_impl, 'supports_groupwise_region', False)) and
            self.rounding_mode != 'STOCHASTIC_ROUND')
        self.region_quant = self.tensor_quant.forward_group if self.supports_quant_weight_region else None
        self.is_region_quant_compiled = False

    def init_tensor_quant(self, preserve_state_dict=False):
        super().init_tensor_quant(preserve_state_dict)
        if hasattr(self, 'region_quant'):
            self._refresh_region_quant()

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

    def quantize_weight_group(self, weight: torch.Tensor,
                              group_index: int) -> Optional[torch.Tensor]:
        if (not self.supports_quant_weight_region or not self.is_quant_enabled or
                self.export_mode or self.training):
            return None
        if weight.ndim != 2 or self.group_dim != 1 or self.region_quant is None:
            return None
        group_start = group_index * self.group_size
        if group_index < 0 or group_start >= weight.shape[self.group_dim]:
            return None
        group, logical_group_size = extract_groupwise_block(
            weight, self.group_dim, self.group_size, group_index)
        quantized_group = self.region_quant(group)[0]
        quantized_group = quantized_group.squeeze(self.group_dim)
        return quantized_group.narrow(self.group_dim, 0, logical_group_size)

    def quantize_weight_region(
            self,
            weight: torch.Tensor,
            bounds: List[Tuple[int, int]],
            quant_input=None) -> Optional[torch.Tensor]:
        if weight.ndim != 2 or len(bounds) != 2:
            return None
        row_start, row_end = bounds[0]
        col_start, col_end = bounds[1]
        if col_start == col_end:
            return weight[row_start:row_end, col_start:col_end]
        group_index = col_start // self.group_size
        group_start = group_index * self.group_size
        group_end = min(group_start + self.group_size, weight.shape[1])
        if col_end > group_end:
            return None
        group = self.quantize_weight_group(weight, group_index)
        if group is None:
            return None
        return group[row_start:row_end, col_start - group_start:col_end - group_start]

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
