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
from torch.nn import Parameter

import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.function.ops import min_int, max_int
from brevitas.utils.python_utils import AutoName
from .restrict_val import RestrictValue, RestrictValueType, FloatToIntImplType, RestrictValueOpImplType
from .stats import StatsOp, StatsInputViewShapeImpl, ParameterListStats, RuntimeStats
from .utils import StatelessBuffer

SCALING_SCALAR_SHAPE = ()


class ScalingImplType(AutoName):
    HE = auto()
    CONST = auto()
    STATS = auto()
    AFFINE_STATS = auto()
    PARAMETER = auto()
    PARAMETER_FROM_STATS = auto()
    OVERRIDE = auto()


class ParameterScaling(torch.jit.ScriptModule):

    def __init__(self,
                 scaling_init: torch.Tensor,
                 parameter_shape: Optional[Tuple[int, ...]],
                 scaling_min_val: Optional[float],
                 restrict_scaling_type: RestrictValueType) -> None:
        super(ParameterScaling, self).__init__()

        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for standalone scaling."
                            .format(str(restrict_scaling_type)))

        self.restrict_value = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL, scaling_min_val)
        scaling_init_op = RestrictValue.restrict_value_op(
            restrict_scaling_type,
            restrict_value_op_impl_type=RestrictValueOpImplType.TORCH_FN)
        scaling_init = scaling_init_op(scaling_init)
        if scaling_init.dim() == 0:  # for activations with per channel scaling
            self.value = Parameter(torch.full(parameter_shape, scaling_init))
        elif scaling_init.shape == parameter_shape:
            self.value = Parameter(scaling_init)  # for weight with per output channel scaling from stats
        else:
            raise Exception("Problem with init of standalone scaling from value {}".format(str(scaling_init)))

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        value = self.restrict_value(self.value)
        return value

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = prefix + 'value'
        retrocomp_value_key = prefix + 'learned_value'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(ParameterScaling, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)


class ConstScaling(torch.jit.ScriptModule):

    def __init__(self,
                 scaling_init: torch.Tensor,
                 scaling_min_val: Optional[float],
                 restrict_scaling_type: RestrictValueType) -> None:
        super(ConstScaling, self).__init__()

        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for standalone scaling."
                            .format(str(restrict_scaling_type)))

        assert scaling_init.dim() == 0
        self.restrict_value = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL, scaling_min_val)
        scaling_init_op = RestrictValue.restrict_value_op(
            restrict_scaling_type,
            restrict_value_op_impl_type=RestrictValueOpImplType.TORCH_FN)
        scaling_init = scaling_init_op(scaling_init)
        self.value = StatelessBuffer(scaling_init)

    @torch.jit.script_method
    def forward(self, placeholder: torch.Tensor) -> torch.Tensor:
        value = self.restrict_value(self.value())
        return value


class AffineRescaling(torch.jit.ScriptModule):

    def __init__(self, affine_shape):
        super(AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(affine_shape))
        self.affine_bias = Parameter(torch.zeros(affine_shape))

    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias
        out = torch.abs(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(AffineRescaling, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        affine_weight_key = prefix + 'affine_weight'
        affine_bias_key = prefix + 'affine_bias'
        if config.IGNORE_MISSING_KEYS and affine_weight_key in missing_keys:
            missing_keys.remove(affine_weight_key)
        if config.IGNORE_MISSING_KEYS and affine_bias_key in missing_keys:
            missing_keys.remove(affine_bias_key)


class StatsScaling(torch.jit.ScriptModule):

    def __init__(self,
                 stats_op: StatsOp,
                 restrict_scaling_type: RestrictValueType,
                 stats_output_shape: Tuple[int, ...],
                 scaling_min_val: Optional[float],
                 affine: bool) -> None:
        super(StatsScaling, self).__init__()

        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for stats scaling."
                            .format(str(restrict_scaling_type)))
        if stats_op == StatsOp.MAX_AVE and stats_output_shape != SCALING_SCALAR_SHAPE:
            raise Exception("Scaling with MAX_AVE stats can't be over output channels.")

        if affine:
            self.affine_rescaling = AffineRescaling(stats_output_shape)
        else:
            self.affine_rescaling = Identity()

        self.restrict_scaling = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL, scaling_min_val)
        self.restrict_scaling_preprocess = RestrictValue.restrict_value_op(restrict_scaling_type,
                                                                           restrict_value_op_impl_type=
                                                                           RestrictValueOpImplType.TORCH_MODULE)

    @torch.jit.script_method
    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        stats = self.affine_rescaling(stats)
        stats = self.restrict_scaling_preprocess(stats)
        stats = self.restrict_scaling(stats)
        return stats


class RuntimeStatsScaling(torch.jit.ScriptModule):

    def __init__(self,
                 stats_op: StatsOp,
                 restrict_scaling_type: RestrictValueType,
                 stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 stats_output_shape: Tuple[int, ...],
                 sigma: Optional[float],
                 scaling_min_val: Optional[float],
                 stats_reduce_dim: Optional[int],
                 stats_permute_dims: Tuple,
                 stats_buffer_momentum: Optional[float],
                 stats_buffer_init: float,
                 affine: bool) -> None:
        super(RuntimeStatsScaling, self).__init__()

        self.runtime_stats = RuntimeStats(stats_op=stats_op,
                                          stats_output_shape=stats_output_shape,
                                          stats_reduce_dim=stats_reduce_dim,
                                          stats_input_view_shape_impl=stats_input_view_shape_impl,
                                          stats_buffer_momentum=stats_buffer_momentum,
                                          stats_buffer_init=stats_buffer_init,
                                          stats_permute_dims=stats_permute_dims,
                                          sigma=sigma)
        self.stats_scaling_impl = StatsScaling(restrict_scaling_type=restrict_scaling_type,
                                               scaling_min_val=scaling_min_val,
                                               affine=affine,
                                               stats_op=stats_op,
                                               stats_output_shape=stats_output_shape)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        stats = self.runtime_stats(x)
        return self.stats_scaling_impl(stats)


class ParameterStatsScaling(torch.jit.ScriptModule):

    def __init__(self,
                 stats_op: StatsOp,
                 restrict_scaling_type: RestrictValueType,
                 stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 stats_output_shape: Tuple[int, ...],
                 stats_input_concat_dim: Optional[int],
                 sigma: Optional[float],
                 scaling_min_val: Optional[float],
                 stats_reduce_dim: Optional[int],
                 tracked_parameter_list: List[torch.nn.Parameter],
                 affine: bool) -> None:
        super(ParameterStatsScaling, self).__init__()

        self.parameter_list_stats = ParameterListStats(stats_op=stats_op,
                                                       stats_output_shape=stats_output_shape,
                                                       stats_reduce_dim=stats_reduce_dim,
                                                       stats_input_view_shape_impl=stats_input_view_shape_impl,
                                                       stats_input_concat_dim=stats_input_concat_dim,
                                                       tracked_parameter_list=tracked_parameter_list,
                                                       sigma=sigma)
        self.stats_scaling_impl = StatsScaling(restrict_scaling_type=restrict_scaling_type,
                                               scaling_min_val=scaling_min_val,
                                               affine=affine,
                                               stats_op=stats_op,
                                               stats_output_shape=stats_output_shape)

    @torch.jit.script_method
    def forward(self, ignored: torch.Tensor):
        stats = self.parameter_list_stats()
        return self.stats_scaling_impl(stats)


class SignedFpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, narrow_range):
        super(SignedFpIntScale, self).__init__()
        self.signed = True
        self.narrow_range = narrow_range

    @torch.jit.script_method
    def forward(self, bit_width):
        return - min_int(self.signed, self.narrow_range, bit_width)


class UnsignedFpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self):
        super(UnsignedFpIntScale, self).__init__()
        self.signed = False

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width)


class PowerOfTwoIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed):
        super(PowerOfTwoIntScale, self).__init__()
        self.signed = signed

    @torch.jit.script_method
    def forward(self, bit_width):
        return max_int(self.signed, bit_width) + 1


class IntScaling(torch.jit.ScriptModule):

    def __init__(self,
                 narrow_range: bool,
                 signed: bool,
                 restrict_scaling_type: RestrictValueType) -> None:
        super(IntScaling, self).__init__()

        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for int scaling."
                            .format(str(restrict_scaling_type)))

        if signed and not restrict_scaling_type == RestrictValueType.POWER_OF_TWO:  # FP or LOG_FP
            self.forward_impl = SignedFpIntScale(narrow_range)
        elif not signed and not restrict_scaling_type == RestrictValueType.POWER_OF_TWO:  # FP or LOG_FP
            self.forward_impl = UnsignedFpIntScale()
        elif restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            self.forward_impl = PowerOfTwoIntScale(signed)
        else:
            raise Exception("Restrict value type {} not recognized".format(restrict_scaling_type))

    @torch.jit.script_method
    def forward(self, bit_width):
        int_scale = self.forward_impl(bit_width)
        return int_scale



