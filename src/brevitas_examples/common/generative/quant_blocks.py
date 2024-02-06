"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper.shape import PermuteDims
from brevitas.core.utils import SliceTensor
from brevitas.core.zero_point import _ScaleShiftZeroPoint
from brevitas.function.ops_ste import abs_binary_sign_grad


class OverSubChannelBlockView(brevitas.jit.ScriptModule):
    __constants__ = ['scaling_input_shape']

    def __init__(self, scaling_input_shape, permute_dims: Optional[Tuple[int, ...]]) -> None:
        super(OverSubChannelBlockView, self).__init__()
        self.scaling_input_shape = scaling_input_shape
        if permute_dims is not None:
            self.permute_impl = PermuteDims(permute_dims)
        else:
            self.permute_impl = nn.Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        y = self.permute_impl(x)
        y = y.view(self.scaling_input_shape)
        return y


class ExpandReshapeScalingWrapper(brevitas.jit.ScriptModule):
    __constants__ = ['expanded_scaling_shape', 'reshaped_scaling_shape']

    def __init__(self, wrapped_scaling_impl, expanded_scaling_shape, reshaped_scaling_shape):
        super(ExpandReshapeScalingWrapper, self).__init__()
        self.wrapped_scaling_impl = wrapped_scaling_impl
        self.expanded_scaling_shape = expanded_scaling_shape
        self.reshaped_scaling_shape = reshaped_scaling_shape
        self.slice_tensor = SliceTensor()

    @brevitas.jit.script_method
    def forward(self, x):
        scale = self.wrapped_scaling_impl(x)
        scale = scale.expand(self.expanded_scaling_shape)
        scale = scale.reshape(self.reshaped_scaling_shape)
        # slice tensor when required by partial quantization
        scale = self.slice_tensor(scale)
        return scale


class ExpandReshapeZeroPointWrapper(brevitas.jit.ScriptModule):
    __constants__ = ['expanded_zero_point_shape', 'reshaped_zero_point_shape']

    def __init__(
            self, wrapped_zero_point_impl, expanded_zero_point_shape, reshaped_zero_point_shape):
        super(ExpandReshapeZeroPointWrapper, self).__init__()
        self.wrapped_zero_point_impl = wrapped_zero_point_impl
        self.expanded_zero_point_shape = expanded_zero_point_shape
        self.reshaped_zero_point_shape = reshaped_zero_point_shape
        self.slice_tensor = SliceTensor()

    def unexpanded_zero_point(self, unexpanded_scale, bit_width):
        """
        This is used at export time.
        """
        zero_point_stats = self.wrapped_zero_point_impl.parameter_list_stats()
        zero_point = self.wrapped_zero_point_impl.scale_shift_zero_point(
            -zero_point_stats, unexpanded_scale, bit_width)
        return zero_point

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor):
        # We have to break into wrapped_zero_point_impl since we need to expand and reshape
        # Before we call into scale_shift_zero_point
        zero_point_stats = self.wrapped_zero_point_impl.parameter_list_stats()
        zero_point_stats = zero_point_stats.expand(self.expanded_zero_point_shape).contiguous()
        # contiguous() above is to avoid an unsafe_view below
        zero_point_stats = zero_point_stats.reshape(self.reshaped_zero_point_shape)
        # slice tensor when required by partial quantization
        zero_point_stats = self.slice_tensor(zero_point_stats)
        zero_point = self.wrapped_zero_point_impl.scale_shift_zero_point(
            -zero_point_stats, scale, bit_width)
        return zero_point


# TODO: restore JIT compatibility
class RuntimeDynamicStatsScaling(nn.Module):

    def __init__(
            self,
            scaling_stats_impl: nn.Module,
            dynamic_scaling_broadcastable_fn: Callable,
            scaling_stats_input_view_shape_impl: nn.Module) -> None:
        super(RuntimeDynamicStatsScaling, self).__init__()
        self.scaling_stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats_impl = scaling_stats_impl
        self.dynamic_scaling_broadcastable_fn = dynamic_scaling_broadcastable_fn

    def forward(self, x) -> Tensor:
        shape = x.shape
        x = self.scaling_stats_input_view_shape_impl(x)
        x = self.stats_impl(x)
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


class RuntimeDynamicGroupStatsScaling(brevitas.jit.ScriptModule):

    def __init__(self, group_size: int, group_dim: int, scaling_stats_impl: nn.Module) -> None:
        super(RuntimeDynamicGroupStatsScaling, self).__init__()
        self.group_size = group_size
        self.group_dim = group_dim
        self.scaling_stats_impl = scaling_stats_impl

    @brevitas.jit.script_method
    def group_scaling_reshape(self, stats_input):
        tensor_shape = stats_input.shape
        tensor_shape_list = list(tensor_shape)
        tensor_shape_list[self.group_dim] = int(tensor_shape_list[self.group_dim] / self.group_size)
        block_dim = self.group_dim + 1 if self.group_dim != -1 else -1
        tensor_shape_list.insert(block_dim, self.group_size)
        stats_input = stats_input.view(tensor_shape_list)
        return stats_input

    @brevitas.jit.script_method
    def forward(self, stats_input) -> Tensor:
        stats_input_reshaped = self.group_scaling_reshape(stats_input)
        out = self.scaling_stats_impl(stats_input_reshaped)
        out = torch.clamp_min(out, min=torch.tensor(1e-6, device=out.device, dtype=out.dtype))
        out = out.expand(stats_input_reshaped.shape)
        out = out.reshape(stats_input.shape)
        return out
