"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable

from torch import Tensor
import torch.nn as nn

from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.zero_point import _ScaleShiftZeroPoint
from brevitas.function.ops_ste import abs_binary_sign_grad


# TODO: restore JIT compatibility
class RuntimeDynamicStatsScaling(nn.Module):

    def __init__(
            self,
            scaling_stats_impl: nn.Module,
            dynamic_scaling_broadcastable_fn: Callable,
            scaling_stats_input_view_shape_impl: nn.Module,
            restrict_scaling_impl: nn.Module,
            restrict_threshold_impl: nn.Module = None,
            scaling_min_val=None) -> None:
        super(RuntimeDynamicStatsScaling, self).__init__()
        # Ensure retro-compatibility with shared threshold/scaling restrict
        if restrict_threshold_impl is None:
            restrict_threshold_impl = restrict_scaling_impl
        self.scaling_stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats_impl = scaling_stats_impl
        self.dynamic_scaling_broadcastable_fn = dynamic_scaling_broadcastable_fn
        self.restrict_scaling_pre = restrict_scaling_impl.restrict_init_module()
        self.restrict_clamp_scaling = _RestrictClampValue(
            scaling_min_val=scaling_min_val, restrict_value_impl=restrict_scaling_impl)
        self.restrict_threshold_pre = restrict_threshold_impl.restrict_init_module()
        self.restrict_clamp_threshold = _RestrictClampValue(
            restrict_value_impl=restrict_threshold_impl)

    def forward(self, x, threshold) -> Tensor:
        shape = x.shape
        threshold = self.restrict_clamp_threshold(self.restrict_threshold_pre(threshold))
        x = self.scaling_stats_input_view_shape_impl(x)
        x = self.stats_impl(x)
        x = self.restrict_clamp_scaling(self.restrict_scaling_pre(x))
        x = x / threshold

        x = self.dynamic_scaling_broadcastable_fn(x, shape)
        return x


# TODO: restore JIT compatibility
class RuntimeDynamicStatsZeroPoint(nn.Module):

    def __init__(
            self,
            zero_point_stats_impl: nn.Module,
            int_quant: nn.Module,
            quantize_zero_point: bool,
            dynamic_scaling_broadcastable_fn: Callable,
            zero_point_stats_input_view_shape_impl: nn.Module) -> None:
        super(RuntimeDynamicStatsZeroPoint, self).__init__()
        self.zero_point_stats_input_view_shape_impl = zero_point_stats_input_view_shape_impl
        self.zero_point_stats_impl = zero_point_stats_impl
        self.dynamic_scaling_broadcastable_fn = dynamic_scaling_broadcastable_fn
        self.scale_shift_zero_point = _ScaleShiftZeroPoint(int_quant, quantize_zero_point)

    def forward(self, x, scale, bit_width) -> Tensor:
        shape = x.shape
        x = self.zero_point_stats_input_view_shape_impl(x)
        x = self.zero_point_stats_impl(x)
        x = self.dynamic_scaling_broadcastable_fn(x, shape)
        x = abs_binary_sign_grad(x)
        x = self.scale_shift_zero_point(x, scale, bit_width)
        return x
