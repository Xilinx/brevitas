# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import this

from brevitas.core.bit_width import *
from brevitas.core.bit_width.float import ComputeMaxMantissa
from brevitas.core.bit_width.float import StaticMaxMantissa
from brevitas.core.function_wrapper import *
from brevitas.core.function_wrapper.learned_round import LearnedRoundHardSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundIdentity
from brevitas.core.function_wrapper.learned_round import LearnedRoundSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.core.function_wrapper.stochastic_round import StochasticRoundSte
from brevitas.core.quant import *
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import *
from brevitas.core.scaling import *
from brevitas.core.scaling import ScalingImplType
from brevitas.core.scaling import ScalingPerOutputType
from brevitas.core.stats import *
from brevitas.function.ops import compute_max_mantissa
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
    'SolveScalingStatsInputViewShapeImplFromEnum',
    'SolveDtypeDeviceFromTrackedParameterList',
    'SolveScaleSignedness']


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
    elif impl_type == FloatToIntImplType.STOCHASTIC_ROUND:
        return StochasticRoundSte
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


class ExponentBitWidthClass(ExtendedInjector):
    exponent_bit_width_impl_type = (this << 1).exponent_bit_width_impl_type
    bit_width = (this << 1).exponent_bit_width

    @value
    def bit_width_impl_type(exponent_bit_width_impl_type):
        return solve_bit_width_impl_from_enum(exponent_bit_width_impl_type)


class MantissaBitWidthClass(ExtendedInjector):
    mantissa_bit_width_impl_type = (this << 1).mantissa_bit_width_impl_type
    bit_width = (this << 1).mantissa_bit_width

    @value
    def bit_width_impl_type(mantissa_bit_width_impl_type):
        return solve_bit_width_impl_from_enum(mantissa_bit_width_impl_type)

    @value
    def compute_max_mantissa(mantissa_bit_width_impl_type, bit_width):
        if mantissa_bit_width_impl_type == BitWidthImplType.CONST or mantissa_bit_width_impl_type == BitWidthImplType.STATEFUL_CONST:
            return StaticMaxMantissa(compute_max_mantissa(torch.tensor(float(bit_width))))
        else:
            return ComputeMaxMantissa


class SolveFloatBitWidthImplFromEnum(ExtendedInjector):

    exponent_bit_class = ExponentBitWidthClass
    mantissa_bit_class = MantissaBitWidthClass

    @value
    def exponent_bit_width_impl():
        return this.exponent_bit_class.bit_width_impl_type

    @value
    def mantissa_bit_width_impl():
        return this.mantissa_bit_class.bit_width_impl_type

    @value
    def compute_max_mantissa():
        return this.mantissa_bit_class.compute_max_mantissa


class SolveBitWidthImplFromEnum(ExtendedInjector):

    @value
    def bit_width_impl(bit_width_impl_type):
        return solve_bit_width_impl_from_enum(bit_width_impl_type)


class SolveScalingStatsOpFromEnum(ExtendedInjector):

    @value
    def scaling_stats_impl(scaling_stats_op=None):
        if scaling_stats_op is None:
            return None
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
        if learned_round_impl_type == LearnedRoundImplType.IDENTITY:
            return LearnedRoundIdentity

    @value
    def learned_round_init(tracked_parameter_list):
        if len(tracked_parameter_list) > 1:
            raise RuntimeError('LearnedRound does not support shared quantizers')
        return torch.full(tracked_parameter_list[0].shape, 0.)


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
    def stats_reduce_dim(scaling_stats_op, scaling_per_output, group_dim=None):
        if scaling_per_output == ScalingPerOutputType.CHANNEL or scaling_stats_op == StatsOp.MAX_AVE:
            return SCALING_STATS_REDUCE_DIM
        elif scaling_per_output == ScalingPerOutputType.TENSOR:
            return None
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            reduce_dim = group_dim + 1 if group_dim >= 0 else group_dim
            return reduce_dim

    @value
    def keepdim(scaling_per_output):
        if scaling_per_output == ScalingPerOutputType.GROUP:
            return True
        else:
            return False

    # Retrocompatibility. Priority is given to scaling_per_output_channel binary flag.
    # We might want to check for discrepancies between the two and raise an error.
    @value
    def scaling_per_output(scaling_per_output_type=None, scaling_per_output_channel=None):
        if scaling_per_output_channel is not None:
            return ScalingPerOutputType.CHANNEL if scaling_per_output_channel else ScalingPerOutputType.TENSOR
        elif scaling_per_output_type is not None:
            return scaling_per_output_type


class SolveScalingStatsInputViewShapeImplFromEnum(ExtendedInjector):

    @value
    def scaling_stats_input_view_shape_impl(scaling_stats_op, scaling_per_output):
        if scaling_per_output == ScalingPerOutputType.CHANNEL or scaling_stats_op == StatsOp.MAX_AVE:
            return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
        elif scaling_per_output == ScalingPerOutputType.TENSOR:
            return StatsInputViewShapeImpl.OVER_TENSOR
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            return StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK

    @value
    def permute_dims(scaling_stats_permute_dims):
        # retrocompatibility with older activation per-channel scaling API
        return scaling_stats_permute_dims


class SolveDtypeDeviceFromTrackedParameterList(ExtendedInjector):

    @value
    def dtype(tracked_parameter_list):
        if len(tracked_parameter_list) > 0:
            return tracked_parameter_list[0].dtype
        else:
            return None

    @value
    def device(tracked_parameter_list):
        if len(tracked_parameter_list) > 0:
            return tracked_parameter_list[0].device
        else:
            return None


class SolveScaleSignedness(ExtendedInjector):

    @value
    def is_scale_unsigned(scaling_stats_impl=None, scaling_init=None):
        if scaling_init is not None:
            if not isinstance(scaling_init, torch.Tensor):
                scaling_init = torch.tensor(scaling_init)

            # Check if any of the init values is negative
            if scaling_init.shape == ():
                is_scale_negative = scaling_init < 0
            else:
                is_scale_negative = any(scaling_init.flatten() < 0)
            return not is_scale_negative
        elif hasattr(scaling_stats_impl, 'is_scale_unsigned'):
            return scaling_stats_impl.is_scale_unsigned
        else:
            # If it is not possible to infer the scale of the sign, we assume it is unsigned
            return True
