"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable

from dependencies import this
from dependencies import value
from torch import Tensor
import torch.nn as nn

from brevitas.core.restrict_val import _RestrictClampValue
# RuntimeDynamicStatsScaling has been promoted to brevitas core (resolved from
# ScalingImplType.DYNAMIC by the act solver); re-exported here for backwards
# compatibility.
from brevitas.core.scaling.runtime import RuntimeDynamicStatsScaling
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import _ScaleShiftZeroPoint
from brevitas.function.ops_ste import abs_binary_sign_grad
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingPerOutputType


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


class QuantScaleScaleShapeMixin(ExtendedInjector):
    """Compute the ``scaling_shape`` of a *quantized-scale* quantizer.

    This mixin is layered onto the nested quantizer that quantizes the scale of
    an outer quantizer. It mirrors the normal per-output scaling shape, but when
    the *outer* (``upstream``) quantizer is groupwise the scale carries one extra
    dimension, so an additional singleton axis is inserted.
    """

    @value
    def scaling_shape(
            scaling_per_output,
            scaling_per_output_channel_shape,
            expanded_groupwise_shape,
            group_dim,
            upstream_scaling):
        if scaling_per_output == ScalingPerOutputType.TENSOR:
            scaling = SCALAR_SHAPE
        elif scaling_per_output == ScalingPerOutputType.CHANNEL:
            scaling = scaling_per_output_channel_shape
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
            assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
            assert group_dim is not None, "Per Group scaling not correctly configured"
            size = list(expanded_groupwise_shape)
            size[group_dim + 1] = 1
            scaling = tuple(size)

        # When quantizing scale of groupwise, there will be one extra dim compared to the normal case
        if upstream_scaling == ScalingPerOutputType.GROUP:
            scaling = list(scaling)
            scaling.insert(-1, 1)
            scaling = tuple(scaling)
        return scaling
