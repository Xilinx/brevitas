# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from enum import auto
from typing import Tuple, Optional, List

import torch
from torch.nn import Parameter, Module

import brevitas.config as config
from brevitas.core.function_wrapper import Identity, ClampMin
from brevitas.function.ops import min_int, max_int
from brevitas.utils.python_utils import AutoName
from .stats import _ParameterListStats, _RuntimeStats
from .utils import StatelessBuffer


SCALING_SCALAR_SHAPE = ()
MOMENTUM = 0.1


class ScalingImplType(AutoName):
    HE = auto()
    CONST = auto()
    STATS = auto()
    AFFINE_STATS = auto()
    PARAMETER = auto()
    PARAMETER_FROM_STATS = auto()
    OVERRIDE = auto()


class _RestrictScaling(torch.jit.ScriptModule):

    def __init__(self, scaling_min_val: Optional[float], restrict_value_impl: Module):
        super(_RestrictScaling, self).__init__()
        self.restrict_value_impl = restrict_value_impl
        if scaling_min_val is not None and scaling_min_val != 0:
            scaling_min_val = restrict_value_impl.restrict_init_float(scaling_min_val)
            self.clamp_min = ClampMin(scaling_min_val) # TODO: should be STE
        else:
            self.clamp_min = Identity()

    def forward(self, x: torch.Tensor):
        x = self.clamp_min(x)
        return self.restrict_value_impl(x)


class _AffineRescaling(torch.jit.ScriptModule):

    def __init__(self, scaling_shape):
        super(_AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(scaling_shape))
        self.affine_bias = Parameter(torch.zeros(scaling_shape))

    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias
        out = torch.abs(out)
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


class _StatsScaling(torch.jit.ScriptModule):

    def __init__(self,
                 scaling_min_val: Optional[float],
                 restrict_value_impl: Module,
                 scaling_shape: Tuple[int, ...],
                 affine: bool) -> None:
        super(_StatsScaling, self).__init__()

        if affine:
            self.affine_rescaling = _AffineRescaling(scaling_shape)
        else:
            self.affine_rescaling = Identity()
        self.restrict_scaling_impl = _RestrictScaling(scaling_min_val, restrict_value_impl)
        self.restrict_scaling_pre = restrict_value_impl.restrict_init_module()

    @torch.jit.script_method
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        stats = self.restrict_scaling_pre(stats)
        stats = self.affine_rescaling(stats)
        stats = self.restrict_scaling_impl(stats)
        return stats


class ParameterScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            scaling_init: torch.Tensor,
            scaling_shape: Tuple[int, ...],
            scaling_min_val: Optional[float],
            restrict_value_impl: Module) -> None:
        super(ParameterScaling, self).__init__()

        self.restrict_scaling_impl = _RestrictScaling(scaling_min_val, restrict_value_impl)
        scaling_init = restrict_value_impl.restrict_init_tensor(scaling_init)
        if scaling_init.dim() == 0:
            self.value = Parameter(torch.full(scaling_shape, scaling_init))
        else:
            assert scaling_init.shape == scaling_shape
            self.value = Parameter(scaling_init)

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        value = self.restrict_scaling_impl(self.value)
        return value

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(ParameterScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ConstScaling(torch.jit.ScriptModule):

    def __init__(self, scaling_init: float) -> None:
        super(ConstScaling, self).__init__()
        self.value = StatelessBuffer(torch.tensor(scaling_init))

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        return self.value()


class RuntimeStatsScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            stats_impl: Module,
            stats_input_view_shape_impl: Module,
            stats_permute_dims: Tuple[int, ...],
            stats_buffer_init: float,
            stats_buffer_momentum: float,
            scaling_min_val: Optional[float],
            restrict_value_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine: bool) -> None:
        super(RuntimeStatsScaling, self).__init__()

        self.runtime_stats = _RuntimeStats(
            stats_impl,
            scaling_shape,
            stats_input_view_shape_impl,
            stats_permute_dims,
            stats_buffer_init,
            stats_buffer_momentum)
        self.stats_scaling_impl = _StatsScaling(
            scaling_min_val,
            restrict_value_impl,
            scaling_shape,
            affine)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        stats = self.runtime_stats(x)
        return self.stats_scaling_impl(stats)


class ParameterStatsScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            stats_impl: Module,
            stats_input_view_shape_impl: Module,
            stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter],
            scaling_min_val: Optional[float],
            restrict_value_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine: bool) -> None:
        super(ParameterStatsScaling, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            stats_impl,
            scaling_shape,
            stats_input_view_shape_impl,
            stats_input_concat_dim,
            tracked_parameter_list)
        self.stats_scaling_impl = _StatsScaling(
            scaling_min_val,
            restrict_value_impl,
            scaling_shape,
            affine)

    @torch.jit.script_method
    def forward(self, ignored: torch.Tensor):
        stats = self.parameter_list_stats()
        return self.stats_scaling_impl(stats)


class FloatIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, signed, narrow_range):
        super(FloatIntScale, self).__init__()
        if not signed and narrow_range:
            raise RuntimeError("Can't have signed narrow range quantization")
        self.signed = signed
        self.narrow_range = narrow_range

    @torch.jit.script_method
    def forward(self, bit_width):
        if self.signed:
            return - min_int(self.signed, self.narrow_range, bit_width)
        else:
            return max_int(self.signed, bit_width)


class PowerOfTwoIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed):
        super(PowerOfTwoIntScale, self).__init__()
        self.signed = signed

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width) + 1



