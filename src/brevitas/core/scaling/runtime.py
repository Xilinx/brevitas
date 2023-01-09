# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple, Optional, List, Union

import torch
from torch.nn import Parameter, Module

import brevitas
import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.function.ops_ste import abs_binary_sign_grad

from brevitas.core.stats import _ParameterListStats, _RuntimeStats, DEFAULT_MOMENTUM
from brevitas.core.restrict_val import _RestrictClampValue


class StatsFromParameterScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter],
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool = False,
            scaling_min_val: Optional[float] = None) -> None:
        super(StatsFromParameterScaling, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_input_concat_dim,
            tracked_parameter_list)
        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl,
            scaling_shape,
            scaling_min_val,
            affine_rescaling)

    @brevitas.jit.script_method
    def forward(self, ignored: torch.Tensor) -> torch.Tensor:
        stats = self.parameter_list_stats()
        return self.stats_scaling_impl(stats)


class _StatsScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            scaling_min_val: Optional[float] = None,
            affine_rescaling: bool = False) -> None:
        super(_StatsScaling, self).__init__()

        if affine_rescaling:
            self.affine_rescaling = _AffineRescaling(scaling_shape)
        else:
            self.affine_rescaling = Identity()
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        self.restrict_scaling_pre = restrict_scaling_impl.restrict_init_module()

    @brevitas.jit.script_method
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        stats = self.restrict_scaling_pre(stats)
        stats = self.affine_rescaling(stats)
        stats = self.restrict_clamp_scaling(stats)
        return stats


class RuntimeStatsScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool,
            scaling_stats_momentum: float = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = None) -> None:
        super(RuntimeStatsScaling, self).__init__()

        self.runtime_stats = _RuntimeStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_momentum)
        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl,
            scaling_shape,
            scaling_min_val,
            affine_rescaling)

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        stats = self.runtime_stats(x)
        return self.stats_scaling_impl(stats)


class _AffineRescaling(brevitas.jit.ScriptModule):

    def __init__(self, scaling_shape):
        super(_AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(scaling_shape))
        self.affine_bias = Parameter(torch.zeros(scaling_shape))

    @brevitas.jit.script_method
    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias
        out = abs_binary_sign_grad(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_AffineRescaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        affine_weight_key = prefix + 'affine_weight'
        affine_bias_key = prefix + 'affine_bias'
        if config.IGNORE_MISSING_KEYS and affine_weight_key in missing_keys:
            missing_keys.remove(affine_weight_key)
        if config.IGNORE_MISSING_KEYS and affine_bias_key in missing_keys:
            missing_keys.remove(affine_bias_key)