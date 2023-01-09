# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from dependencies import this, value

from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingImplType, StatsOp, RestrictValueType
from brevitas.inject.enum import QuantType, BitWidthImplType, FloatToIntImplType
from brevitas.core.zero_point import ZeroZeroPoint, StatsFromParameterZeroPoint
from brevitas.core.zero_point import ParameterFromRuntimeZeroPoint
from brevitas.core.quant import ClampedBinaryQuant
from brevitas.core.scaling import IntScaling, ParameterScaling, StatsFromParameterScaling
from brevitas.core.scaling import SCALING_STATS_REDUCE_DIM, SCALAR_SHAPE
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.stats import AbsMax, AbsMaxL2
from brevitas.core.stats import NegativeMinOrZero, NegativePercentileOrZero
from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant.int import DecoupledRescalingIntQuant
from brevitas.core.quant.int_base import DecoupledIntQuant
from brevitas.core.function_wrapper import TensorClampSte
from brevitas.core.function_wrapper import OverOutputChannelView
from brevitas.quant.solver.parameter import ParameterFromStatsScalingInit
from brevitas.quant.solver.weight import SolveWeightScalingStatsInputDimsFromModule
from brevitas.quant.solver.weight import SolveWeightScalingPerOutputChannelShapeFromModule#
from brevitas.quant.solver.parameter import SolveParameterScalingShape
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector

__all__ = [
    'MaxStatsScaling',
    'MinMaxStatsScaling',
    'ParamFromRuntimePercentileScaling',
    'ParamFromRuntimePercentileIntervalScaling',
    'ParamFromRuntimeMinMaxScaling',
    'ParamMinMaxInitScaling',
    'IntQuant',
    'NarrowIntQuant',
    'UintQuant',
    'ShiftedMinUintQuant',
    'ShiftedParamFromPercentileUintQuant',
    'PerChannelFloatScaling8bit',
    'PerTensorFloatScaling8bit',
    'PerTensorPoTScaling8bit',
    'IntTrunc',
    'SignedBinaryClampedConst',
    'WeightPerTensorFloatDecoupledL2Param',
    'WeightPerChannelFloatDecoupled'
]


class MaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_min_val = 1e-10


class MinMaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MIN_MAX
    scaling_min_val = 1e-10


class ParamFromRuntimePercentileScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    high_percentile_q = 99.999
    collect_stats_steps = 300
    scaling_min_val = 1e-10


class ParamFromRuntimeMinMaxScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.MIN_MAX
    collect_stats_steps = 300
    scaling_min_val = 1e-10
    
    
class ParamFromRuntimePercentileIntervalScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE_INTERVAL
    high_percentile_q = 99.999
    low_percentile_q = 0.001
    collect_stats_steps = 300
    scaling_min_val = 1e-10


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
    quantize_zero_point = True
    zero_point_impl = StatsFromParameterZeroPoint
    zero_point_stats_impl = NegativeMinOrZero
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
    zero_point_stats_input_concat_dim = this.scaling_stats_input_concat_dim


class ShiftedParamFromPercentileUintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    quantize_zero_point = True
    zero_point_impl = ParameterFromRuntimeZeroPoint
    zero_point_stats_impl = NegativePercentileOrZero
    low_percentile_q = 0.001
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
    restrict_value_float_to_int_impl = CeilSte


class IntTrunc(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.FLOOR


class SignedBinaryClampedConst(ExtendedInjector):
    tensor_quant = ClampedBinaryQuant
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True


class PerTensorConstScaling2bit(ExtendedInjector):
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    bit_width = 2


class WeightPerTensorFloatDecoupledL2Param(SolveWeightScalingStatsInputDimsFromModule):
    """
    Experimental narrow per-tensor signed int weight quantizer fragment with decoupled L2,inf
    normalization and learned scaling.
    """

    @value
    def scaling_init(scaling_init_impl):
        return scaling_init_impl()

    proxy_class = DecoupledWeightQuantProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuant
    decoupled_int_quant = DecoupledIntQuant
    tensor_clamp_impl = TensorClampSte
    pre_scaling_impl = StatsFromParameterScaling
    scaling_stats_impl = AbsMaxL2
    scaling_stats_input_view_shape_impl = OverOutputChannelView
    scaling_stats_input_concat_dim = 0
    stats_reduce_dim = SCALING_STATS_REDUCE_DIM
    restrict_scaling_impl = FloatRestrictValue
    scaling_shape = SCALAR_SHAPE
    scaling_impl = ParameterScaling
    scaling_init_impl = ParameterFromStatsScalingInit
    parameter_stats_scaling_init_impl = this.pre_scaling_impl
    int_scaling_impl = IntScaling
    zero_point_impl = ZeroZeroPoint
    pre_zero_point_impl = ZeroZeroPoint
    bit_width_impl = BitWidthConst
    narrow_range = True
    signed = True
    
    
class WeightPerChannelFloatDecoupled(
        SolveWeightScalingStatsInputDimsFromModule,
        SolveWeightScalingPerOutputChannelShapeFromModule,
        SolveParameterScalingShape):
    """
    Experimental narrow per-channel signed int weight quantizer fragment with decoupled Linf
    normalization and learned scaling.
    """

    @value
    def scaling_init(scaling_init_impl):
        return scaling_init_impl()

    proxy_class = DecoupledWeightQuantProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuant
    decoupled_int_quant = DecoupledIntQuant
    tensor_clamp_impl = TensorClampSte
    pre_scaling_impl = StatsFromParameterScaling
    scaling_stats_impl = AbsMax
    restrict_scaling_impl = FloatRestrictValue
    scaling_impl = ParameterScaling
    scaling_init_impl = ParameterFromStatsScalingInit
    parameter_stats_scaling_init_impl = this.pre_scaling_impl
    int_scaling_impl = IntScaling
    zero_point_impl = ZeroZeroPoint
    pre_zero_point_impl = ZeroZeroPoint
    bit_width_impl = BitWidthConst
    narrow_range = True
    signed = True
    scaling_stats_input_view_shape_impl = OverOutputChannelView
    stats_reduce_dim = SCALING_STATS_REDUCE_DIM
    scaling_per_output_channel = True






