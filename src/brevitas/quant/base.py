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


from dependencies import this

from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingImplType, StatsOp, RestrictValueType
from brevitas.inject.enum import QuantType, BitWidthImplType, FloatToIntImplType
from brevitas.core.zero_point import ZeroZeroPoint, MinUintZeroPoint
from brevitas.core.zero_point import ParameterFromRuntimeMinZeroPoint


__all__ = [
    'MaxStatsScaling',
    'MinMaxStatsScaling',
    'ParamFromRuntimePercentileScaling',
    'ParamFromRuntimeMinMaxScaling',
    'ParamMinMaxInitScaling',
    'IntQuant',
    'NarrowIntQuant',
    'UintQuant',
    'ShiftedMinUintQuant',
    'ShiftedRuntimeMinToUintQuant',
    'PerChannelFloatScaling8bit',
    'PerTensorFloatScaling8bit',
    'PerTensorPoTScaling8bit',
    'IntTrunc'
]


class MaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX


class MinMaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MIN_MAX


class ParamFromRuntimePercentileScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30


class ParamFromRuntimeMinMaxScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.MIN_MAX
    collect_stats_steps = 30


class ParamMinMaxInitScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER


class IntQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = True
    zero_point_impl = ZeroZeroPoint


class NarrowIntQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint


class UintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint


class ShiftedMinUintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = MinUintZeroPoint
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl


class ShiftedRuntimeMinToUintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ParameterFromRuntimeMinZeroPoint
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl


class PerChannelFloatScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorFloatScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorPoTScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8


class IntTrunc(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.FLOOR






