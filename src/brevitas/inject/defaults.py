from . import BaseInjector as Injector
from .enum import ScalingImplType, StatsOp, RestrictValueType
from .enum import QuantType, BitWidthImplType, FloatToIntImplType
from brevitas.core.zero_point import ZeroZeroPoint, ShiftToUnsignedZeroPoint


class StatsMaxScaling(Injector):
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX


class ParamFromRuntimePercentileScaling(Injector):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 30


class ParamMinMaxInitScaling(Injector):
    scaling_impl_type = ScalingImplType.PARAMETER


class IntQuant(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = True
    zero_point_impl = ZeroZeroPoint


class NarrowIntQuant(IntQuant):
    narrow_range = True


class UintQuant(IntQuant):
    signed = False


class ShiftedUintQuant(UintQuant):
    zero_point_impl = ShiftToUnsignedZeroPoint


class PerChannelFloatScaling8bit(Injector):
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorFloatScaling8bit(Injector):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorPoTScaling8bit(Injector):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8


class IntTrunc(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.FLOOR


class FloatBias(Injector):
    quant_type = QuantType.FP
    narrow_range = False
    signed = True


TruncTo8bit = IntTrunc.let(bit_width=8)
Int8Bias = IntQuant.let(bit_width=8)
Int8BiasPerTensorFloatInternalScaling = IntQuant & StatsMaxScaling & PerTensorFloatScaling8bit
Int8WeightPerTensorFloat = NarrowIntQuant & StatsMaxScaling & PerTensorFloatScaling8bit
ShiftedUint8WeightPerTensorFloat = ShiftedUintQuant & StatsMaxScaling & PerTensorFloatScaling8bit
Int8ActPerTensorFloat = IntQuant & ParamFromRuntimePercentileScaling & PerTensorFloatScaling8bit
Uint8ActPerTensorFloat = UintQuant & ParamFromRuntimePercentileScaling & PerTensorFloatScaling8bit
ShiftedUint8ActPerTensorFloat = ShiftedUintQuant & ParamFromRuntimePercentileScaling & PerTensorFloatScaling8bit
Int8ActPerTensorFloatMinMaxInit = IntQuant & ParamMinMaxInitScaling & PerTensorFloatScaling8bit
