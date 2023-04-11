# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.bit_width import *
from brevitas.core.function_wrapper import *
from brevitas.core.function_wrapper.learned_round import LearnedRoundHardSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.core.quant import *
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import *
from brevitas.core.scaling import *
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import *
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import LearnedRoundImplType

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
    'SolveScalingStatsInputViewShapeImplFromEnum']


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
    elif impl_type == FloatToIntImplType.LEARNED_ROUND:
        return LearnedRoundSte
    else:
        raise Exception(f"{impl_type} not recognized.")


def solve_bit_width_impl_from_enum(impl_type):
    if impl_type == BitWidthImplType.CONST:
        return BitWidthConst
    elif impl_type == BitWidthImplType.PARAMETER:
        return BitWidthParameter
    elif impl_type == BitWidthImplType.STATEFUL_CONST:
        return BitWidthStatefulConst
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
        elif scaling_stats_op == StatsOp.PERCENTILE_INTERVAL:
            return PercentileInterval
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

    @value
    def learned_round_impl(learned_round_impl_type):
        if learned_round_impl_type == LearnedRoundImplType.SIGMOID:
            return LearnedRoundSigmoid
        if learned_round_impl_type == LearnedRoundImplType.HARD_SIGMOID:
            return LearnedRoundHardSigmoid

    @value
    def learned_round_init(tracked_parameter_list):
        if len(tracked_parameter_list) > 1:
            raise RuntimeError('LearnedRound does not support shared quantizers')
        return torch.full(tracked_parameter_list[0].shape, 0.)


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
