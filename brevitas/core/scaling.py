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
from typing import Callable, Tuple, Optional, List

import torch
from torch.nn import Module, Parameter

from brevitas.utils.python_utils import AutoName
from brevitas.function.ops import min_int, max_int
from .stats import StatsOp, StatsInputViewShapeImpl, ParameterListStats
from .restrict_val import RestrictValue, RestrictValueType, FloatToIntImplType, RestrictValueOpImplType

SCALING_SCALAR_SHAPE = ()


class ScalingImplType(AutoName):
    CONST = auto()
    STATS = auto()
    PARAMETER = auto()
    PARAMETER_FROM_STATS = auto()


class StandaloneScaling(torch.jit.ScriptModule):

    def __init__(self,
                 scaling_init: float,
                 is_parameter: bool,
                 parameter_shape: Optional[Tuple[int, ...]],
                 restrict_scaling_type: RestrictValueType) -> None:
        super(StandaloneScaling, self).__init__()

        if len(parameter_shape) > 1 and not is_parameter:
            raise Exception("Standalone scaling shape has to be a scalar when scaling is not learned.")
        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for standalone scaling."
                            .format(str(restrict_scaling_type)))

        self.restrict_value = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL)
        scaling_init_op = RestrictValue.restrict_value_op(restrict_scaling_type,
                                                          restrict_value_op_impl_type=RestrictValueOpImplType.MATH)
        scaling_init = scaling_init_op(scaling_init)
        if is_parameter:
            self.value = Parameter(torch.full(parameter_shape, scaling_init))
        else:
            self.value = torch.tensor(scaling_init)

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel: torch.Tensor) -> torch.Tensor:
        value = self.value + zero_hw_sentinel
        value = self.restrict_value(value)
        return value

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(StandaloneScaling, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if value_key in missing_keys:
            missing_keys.remove(value_key)


class StatsScaling(torch.jit.ScriptModule):

    def __init__(self,
                 stats_op: StatsOp,
                 restrict_scaling_type: RestrictValueType,
                 tracked_parameter_list: List[torch.nn.Parameter],
                 stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 stats_output_shape: Tuple[int, ...],
                 stats_input_concat_dim: Optional[int],
                 sigma: Optional[float],
                 stats_reduce_dim: Optional[int]) -> None:
        super(StatsScaling, self).__init__()

        if not (restrict_scaling_type == RestrictValueType.FP
                or restrict_scaling_type == RestrictValueType.LOG_FP
                or restrict_scaling_type == RestrictValueType.POWER_OF_TWO):
            raise Exception("Restriction of type {} is not supported for stats scaling."
                            .format(str(restrict_scaling_type)))
        if stats_op == StatsOp.MAX_AVE and stats_reduce_dim is not None:
            raise Exception("Scaling with MAX_AVE stats can't be over output channels.")

        self.restrict_scaling = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL)
        self.restrict_scaling_preprocess = RestrictValue.restrict_value_op(restrict_scaling_type,
                                                                           restrict_value_op_impl_type=
                                                                           RestrictValueOpImplType.TORCH_MODULE)
        self.parameter_list_stats = ParameterListStats(stats_op=stats_op,
                                                       stats_output_shape=stats_output_shape,
                                                       stats_reduce_dim=stats_reduce_dim,
                                                       stats_input_view_shape_impl=stats_input_view_shape_impl,
                                                       tracked_parameter_list=tracked_parameter_list,
                                                       stats_input_concat_dim=stats_input_concat_dim,
                                                       sigma=sigma)

    @torch.jit.script_method
    def forward(self, zero_hw_sentinel: torch.Tensor) -> torch.Tensor:
        stats = self.parameter_list_stats()
        stats = self.restrict_scaling_preprocess(stats)
        stats = self.restrict_scaling(stats)
        return stats


class FpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, signed, narrow_range):
        super(FpIntScale, self).__init__()
        self.signed = signed
        self.narrow_range = narrow_range

    @torch.jit.script_method
    def forward(self, bit_width):
        return - min_int(self.signed, self.narrow_range, bit_width)


class LogFpIntScale(torch.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed):
        super(LogFpIntScale, self).__init__()
        self.signed = signed

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
            self.forward_impl = FpIntScale(signed, narrow_range)
        elif not signed and not restrict_scaling_type == RestrictValueType.POWER_OF_TWO:  # FP or LOG_FP
            self.forward_impl = LogFpIntScale(signed)
        elif restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            self.forward_impl = PowerOfTwoIntScale(signed)
        else:
            raise Exception("Restrict value type {} not recognized".format(restrict_scaling_type))

    @torch.jit.script_method
    def forward(self, bit_width):
        int_scale = self.forward_impl(bit_width)
        return int_scale



