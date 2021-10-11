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


from brevitas.core.quant import *
from brevitas.core.function_wrapper import *
from brevitas.core.scaling import *
from brevitas.core.restrict_val import *
from brevitas.core.bit_width import *
from brevitas.core.quant import QuantType
from brevitas.core.stats import *
from brevitas.core.scaling import ScalingImplType
from brevitas.inject import ExtendedInjector, value


__all__ = [
    'solve_bit_width_impl_from_enum',
    'solve_restrict_value_impl_from_enum',
    'solve_float_to_int_impl_from_enum',
    'SolveAffineRescalingFromEnum',
    'SolveIntQuantFromEnum',
    'SolveTensorQuantFloatToIntImplFromEnum',
    'SolveIntScalingImplFromEnum',
    'SolveRestrictScalingImplFromEnum',
    'SolveScalingStatsOpFromEnum',
    'SolveBitWidthImplFromEnum',
    'SolveStatsReduceDimFromEnum',
    'SolveScalingStatsInputViewShapeImplFromEnum'
]


def solve_float_to_int_impl_from_enum(impl_type):
    if impl_type == FloatToIntImplType.ROUND:
        return RoundSte
    elif impl_type == FloatToIntImplType.FLOOR:
        return FloorSte
    elif impl_type == FloatToIntImplType.CEIL:
        return CeilSte
    elif impl_type == FloatToIntImplType.ROUND_TO_ZERO:
        return RoundToZeroSte
    elif impl_type == FloatToIntImplType.DPU:
        return DPURoundSte
    else:
        raise Exception(f"{impl_type} not recognized.")


def solve_bit_width_impl_from_enum(impl_type):
    if impl_type == BitWidthImplType.CONST:
        return BitWidthConst
    elif impl_type == BitWidthImplType.PARAMETER:
        return BitWidthParameter
    else:
        raise Exception(f"{impl_type} not recognized.")


def solve_restrict_value_impl_from_enum(impl_type):
    if impl_type == RestrictValueType.FP:
        return FloatRestrictValue
    elif impl_type == RestrictValueType.LOG_FP:
        return LogFloatRestrictValue
    elif impl_type == RestrictValueType.POWER_OF_TWO:
        return PowerOfTwoRestrictValue
    else:
        raise RuntimeError(f"{impl_type} not recognized.")


class SolveRestrictScalingImplFromEnum(ExtendedInjector):

    @value
    def restrict_scaling_impl(restrict_scaling_type):
        return solve_restrict_value_impl_from_enum(restrict_scaling_type)


class SolveBitWidthImplFromEnum(ExtendedInjector):

    @value
    def bit_width_impl(bit_width_impl_type):
        return solve_bit_width_impl_from_enum(bit_width_impl_type)


class SolveScalingStatsOpFromEnum(ExtendedInjector):

    @value
    def scaling_stats_impl(scaling_stats_op):
        if scaling_stats_op == StatsOp.MAX:
            return AbsMax
        elif scaling_stats_op == StatsOp.MAX_AVE:
            return AbsMaxAve
        elif scaling_stats_op == StatsOp.AVE:
            return AbsAve
        elif scaling_stats_op == StatsOp.MEAN_SIGMA_STD:
            return MeanSigmaStd
        elif scaling_stats_op == StatsOp.MEAN_LEARN_SIGMA_STD:
            return MeanLearnedSigmaStd
        elif scaling_stats_op == StatsOp.PERCENTILE:
            return AbsPercentile
        elif scaling_stats_op == StatsOp.MIN_MAX:
            return AbsMinMax
        else:
            raise RuntimeError(f"{scaling_stats_op} not recognized.")


class SolveAffineRescalingFromEnum(ExtendedInjector):

    @value
    def affine_rescaling(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.STATS:
            return False
        elif scaling_impl_type == ScalingImplType.AFFINE_STATS:
            return True
        else:
            return None


class SolveIntQuantFromEnum(ExtendedInjector):

    @value
    def int_quant(quant_type):
        if quant_type == QuantType.INT:
            return IntQuant
        else:
            return None


class SolveTensorQuantFloatToIntImplFromEnum(ExtendedInjector):

    @value
    def float_to_int_impl(float_to_int_impl_type):
        return solve_float_to_int_impl_from_enum(float_to_int_impl_type)


class SolveIntScalingImplFromEnum(ExtendedInjector):

    @value
    def int_scaling_impl(restrict_scaling_type):
        if restrict_scaling_type == RestrictValueType.FP:
            return IntScaling
        elif restrict_scaling_type == RestrictValueType.LOG_FP:
            return IntScaling
        elif restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            return PowerOfTwoIntScaling
        else:
            raise RuntimeError(f"{restrict_scaling_type} not recognized.")
    
    
class SolveStatsReduceDimFromEnum(ExtendedInjector):

    @value
    def stats_reduce_dim(scaling_stats_op, scaling_per_output_channel):
        if scaling_stats_op == StatsOp.MAX_AVE or scaling_per_output_channel:
            return SCALING_STATS_REDUCE_DIM
        else:
            return None


class SolveScalingStatsInputViewShapeImplFromEnum(ExtendedInjector):

    @value
    def scaling_stats_input_view_shape_impl(scaling_per_output_channel, scaling_stats_op):
        if scaling_per_output_channel or scaling_stats_op == StatsOp.MAX_AVE:
            return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
        else:
            return StatsInputViewShapeImpl.OVER_TENSOR

    @value
    def permute_dims(scaling_stats_permute_dims):
        # retrocompatibility with older activation per-channel scaling API
        return scaling_stats_permute_dims
