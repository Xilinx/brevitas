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
