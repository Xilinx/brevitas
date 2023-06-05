"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper.shape import PermuteDims
from brevitas.core.utils import StatelessBuffer


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


class AbsMaxKeepDim(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim) -> None:
        super(AbsMaxKeepDim, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is not None:
            y = torch.max(torch.abs(x), dim=self.stats_reduce_dim, keepdim=True)[0]
        else:
            y = torch.max(torch.abs(x))
        return y


class AbsMinMaxKeepDim(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(AbsMinMaxKeepDim, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            return torch.abs(torch.max(x) - torch.min(x))
        else:
            max_val = torch.max(x, dim=self.stats_reduce_dim, keepdim=True)[0]
            min_val = torch.min(x, dim=self.stats_reduce_dim, keepdim=True)[0]
            return torch.abs(max_val - min_val)


class NegativeMinOrZeroKeepDim(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(NegativeMinOrZeroKeepDim, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.zero = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            min_val = torch.min(x, keepdim=True)
        else:
            min_val = torch.min(x, dim=self.stats_reduce_dim, keepdim=True)[0]
        min_val = torch.where(
            min_val <= self.zero().to(min_val.dtype), min_val, self.zero().to(min_val.dtype))
        return min_val


class ExpandReshapeScalingWrapper(brevitas.jit.ScriptModule):
    __constants__ = ['expanded_scaling_shape', 'reshaped_scaling_shape']

    def __init__(self, wrapped_scaling_impl, expanded_scaling_shape, reshaped_scaling_shape):
        super(ExpandReshapeScalingWrapper, self).__init__()
        self.wrapped_scaling_impl = wrapped_scaling_impl
        self.expanded_scaling_shape = expanded_scaling_shape
        self.reshaped_scaling_shape = reshaped_scaling_shape

    def forward(self, x):
        scale = self.wrapped_scaling_impl(x)
        scale = scale.expand(self.expanded_scaling_shape).contiguous()
        # contiguous() above is to avoid an unsafe_view below
        scale = scale.view(self.reshaped_scaling_shape)
        return scale


class ExpandReshapeZeroPointWrapper(brevitas.jit.ScriptModule):
    __constants__ = ['expanded_zero_point_shape', 'reshaped_zero_point_shape']

    def __init__(
            self, wrapped_zero_point_impl, expanded_zero_point_shape, reshaped_zero_point_shape):
        super(ExpandReshapeZeroPointWrapper, self).__init__()
        self.wrapped_zero_point_impl = wrapped_zero_point_impl
        self.expanded_zero_point_shape = expanded_zero_point_shape
        self.reshaped_zero_point_shape = reshaped_zero_point_shape

    def unexpanded_zero_point(self, unexpanded_scale, bit_width):
        """
        This is used at export time.
        """
        zero_point_stats = self.wrapped_zero_point_impl.parameter_list_stats()
        zero_point = self.wrapped_zero_point_impl.scale_shift_zero_point(
            -zero_point_stats, unexpanded_scale, bit_width)
        return zero_point

    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor):
        # We have to break into wrapped_zero_point_impl since we need to expand and reshape
        # Before we call into scale_shift_zero_point
        zero_point_stats = self.wrapped_zero_point_impl.parameter_list_stats()
        zero_point_stats = zero_point_stats.expand(self.expanded_zero_point_shape).contiguous()
        # contiguous() above is to avoid an unsafe_view below
        zero_point_stats = zero_point_stats.view(self.reshaped_zero_point_shape)
        zero_point = self.wrapped_zero_point_impl.scale_shift_zero_point(
            -zero_point_stats, scale, bit_width)
        return zero_point
