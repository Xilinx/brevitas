from dependencies import this

from . import BaseInjector as Injector
from .enum import ScalingImplType, StatsOp, RestrictValueType
from .enum import QuantType, BitWidthImplType, FloatToIntImplType
from brevitas.core.zero_point import ZeroZeroPoint, MinUintZeroPoint
from brevitas.core.zero_point import ParameterFromRuntimeMinZeroPoint
from brevitas.core.stats import AbsMinMax


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
    'ShiftedIntToUintQuant',
    'PerChannelFloatScaling8bit',
    'PerTensorFloatScaling8bit',
    'PerTensorPoTScaling8bit',
    'IntTrunc'
]


class MaxStatsScaling(Injector):
    """

    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX


class MinMaxStatsScaling(Injector):
    """

    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_impl = AbsMinMax


class ParamFromRuntimePercentileScaling(Injector):
    """

    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30


class ParamFromRuntimeMinMaxScaling(Injector):
    """

    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_impl = AbsMinMax
    collect_stats_steps = 30


class ParamMinMaxInitScaling(Injector):
    """

    """
    scaling_impl_type = ScalingImplType.PARAMETER


class IntQuant(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = True
    zero_point_impl = ZeroZeroPoint


class NarrowIntQuant(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint


class UintQuant(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint


class ShiftedMinUintQuant(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = False
    zero_point_impl = MinUintZeroPoint


class ShiftedIntToUintQuant(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = False
    zero_point_impl = ParameterFromRuntimeMinZeroPoint
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl


class PerChannelFloatScaling8bit(Injector):
    """

    """
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorFloatScaling8bit(Injector):
    """

    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorPoTScaling8bit(Injector):
    """

    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8


class IntTrunc(Injector):
    """

    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.FLOOR



