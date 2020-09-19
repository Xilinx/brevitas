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

from typing import Tuple, Optional, List, Union

import torch
from torch.nn import Parameter, Module

import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.function.ops import min_int, max_int
from brevitas.inject.enum import ScalingImplType  # retrocompatibility

from .stats import _ParameterListStats, _RuntimeStats, _Stats, SCALAR_SHAPE
from .utils import StatelessBuffer
from .restrict_val import _RestrictClampValue

SCALING_STATS_REDUCE_DIM = 1
DEFAULT_MOMENTUM = 0.1
DEFAULT_AFFINE = False
DEFAULT_SCALING_MIN_VAL = None

assert ScalingImplType  # prevent removal of unused import


class _AffineRescaling(torch.jit.ScriptModule):

    def __init__(self, scaling_shape):
        super(_AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(scaling_shape))
        self.affine_bias = Parameter(torch.zeros(scaling_shape))

    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias  # TODO: take absvals
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

    def __init__(
            self,
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            scaling_min_val: Optional[float] = DEFAULT_SCALING_MIN_VAL,
            affine_rescaling: bool = DEFAULT_AFFINE) -> None:
        super(_StatsScaling, self).__init__()

        if affine_rescaling:
            self.affine_rescaling = _AffineRescaling(scaling_shape)
        else:
            self.affine_rescaling = Identity()
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        self.restrict_scaling_pre = restrict_scaling_impl.restrict_init_module()

    @torch.jit.script_method
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        stats = self.affine_rescaling(stats)  # TODO it should be first prerestrict then affine
        stats = self.restrict_scaling_pre(stats)
        stats = self.restrict_clamp_scaling(stats)
        return stats


class ParameterScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            scaling_init: Union[float, torch.Tensor],
            scaling_shape: Tuple[int, ...],
            restrict_scaling_impl: Module,
            scaling_min_val: Optional[float] = DEFAULT_SCALING_MIN_VAL) -> None:
        super(ParameterScaling, self).__init__()

        if isinstance(scaling_init, torch.Tensor):
            scaling_init = scaling_init.detach()
        else:
            self.value = torch.tensor(scaling_init)
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
        if scaling_init.dim() == 0:
            self.value = Parameter(torch.full(scaling_shape, scaling_init))
        else:
            assert scaling_init.shape == scaling_shape
            self.value = Parameter(scaling_init)

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        value = self.restrict_clamp_scaling(self.value)
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

    def __init__(
            self,
            scaling_init: Union[float, torch.Tensor],
            restrict_scaling_impl: Module,
            scaling_min_val: Optional[float] = None) -> None:
        super(ConstScaling, self).__init__()
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        if isinstance(scaling_init, torch.Tensor):
            scaling_init = restrict_scaling_impl.restrict_init_tensor(scaling_init)
            self.value = StatelessBuffer(scaling_init.detach())
        else:
            scaling_init = restrict_scaling_impl.restrict_init_float(scaling_init)
            self.value = StatelessBuffer(torch.tensor(scaling_init))

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        value = self.value()
        restricted_value = self.restrict_clamp_scaling(value)
        return restricted_value


class RuntimeStatsScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_permute_dims: Tuple[int, ...],
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool,
            scaling_stats_momentum: float = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = DEFAULT_SCALING_MIN_VAL) -> None:
        super(RuntimeStatsScaling, self).__init__()

        self.runtime_stats = _RuntimeStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_permute_dims,
            scaling_stats_momentum)
        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl,
            scaling_shape,
            scaling_min_val,
            affine_rescaling)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        stats = self.runtime_stats(x)
        return self.stats_scaling_impl(stats)


class ParameterStatsScaling(torch.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter],
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool,
            scaling_min_val: Optional[float] = DEFAULT_SCALING_MIN_VAL) -> None:
        super(ParameterStatsScaling, self).__init__()
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

    @torch.jit.script_method
    def forward(self, ignored: torch.Tensor):
        stats = self.parameter_list_stats()
        return self.stats_scaling_impl(stats)


class ParameterFromRuntimeStatsScaling(torch.jit.ScriptModule):
    __constants__ = ['stats_permute_dims',
                     'collect_stats_steps',
                     'momentum']

    def __init__(
            self,
            collect_stats_steps: int,
            restrict_scaling_impl: Module,
            scaling_stats_impl: Module,
            scaling_shape: Tuple[int, ...],
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_permute_dims: Optional[Tuple[int, ...]],
            scaling_stats_momentum: float = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = DEFAULT_SCALING_MIN_VAL) -> None:
        super(ParameterFromRuntimeStatsScaling, self).__init__()
        assert collect_stats_steps > 0, 'Steps should be more than 0'
        if scaling_shape != SCALAR_SHAPE and scaling_stats_permute_dims is None:
            raise RuntimeError("Per channel runtime stats require a permute shape")
        self.collect_stats_steps = collect_stats_steps
        self.counter: int = torch.jit.Attribute(0, int)
        self.stats_permute_dims = scaling_stats_permute_dims
        self.stats_input_view_shape_impl = scaling_stats_input_view_shape_impl
        self.stats = _Stats(scaling_stats_impl, scaling_shape)
        self.momentum = scaling_stats_momentum
        self.value = Parameter(torch.full(scaling_shape, 1.0))
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        self.restrict_preprocess = restrict_scaling_impl.restrict_init_module()

    @torch.jit.script_method
    def forward(self, stats_input) -> torch.Tensor:
        out = self.value
        if self.training:
            if self.counter < self.collect_stats_steps:
                if self.stats_permute_dims is not None:
                    stats_input = stats_input.permute(*self.stats_permute_dims).contiguous()
                stats_input = self.stats_input_view_shape_impl(stats_input)
                stats = self.stats(stats_input)
                stats = self.restrict_preprocess(stats)
                if self.counter == 0:
                    self.value.detach().mul_(stats.detach())
                else:
                    self.value.detach().mul_(1 - self.momentum)
                    self.value.detach().add_(self.momentum * stats.detach())
                out = stats
                self.counter = self.counter + 1
        out = self.restrict_clamp_scaling(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ParameterFromRuntimeStatsScaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class FloatIntScaling(torch.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, signed, narrow_range):
        super(FloatIntScaling, self).__init__()
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


class PowerOfTwoIntScaling(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed):
        super(PowerOfTwoIntScaling, self).__init__()
        self.signed = signed

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width) + 1
