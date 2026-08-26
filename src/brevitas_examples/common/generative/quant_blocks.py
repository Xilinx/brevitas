"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable

from torch import Tensor
import torch.nn as nn

from brevitas.core.restrict_val import _RestrictClampValue
# RuntimeDynamicStatsScaling has been promoted to brevitas core (resolved from
# ScalingImplType.DYNAMIC by the act solver); re-exported here for backwards
# compatibility.
from brevitas.core.scaling.runtime import RuntimeDynamicStatsScaling
from brevitas.core.zero_point import _ScaleShiftZeroPoint
from brevitas.function.ops_ste import abs_binary_sign_grad


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
