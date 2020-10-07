from . import BaseInjector as Injector
from .enum import ScalingImplType, StatsOp, RestrictValueType
from .enum import QuantType, BitWidthImplType, FloatToIntImplType


class StatsMaxScaling(Injector):
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_min_val = 2.0 ** (-16)


class ParamFromRuntimePercentileScaling(Injector):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    percentile_q = 99.999
    collect_stats_steps = 100


class PerTensorPoTScaling(Injector):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False


class PerTensorFloatScaling(Injector):
    restrict_scaling_type = RestrictValueType.LOG_FP
    scaling_per_output_channel = False


class PerChannelPoTScaling(Injector):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = True


class PerChannelFloatScaling(Injector):
    restrict_scaling_type = RestrictValueType.LOG_FP
    scaling_per_output_channel = True


class Int8(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = True
    bit_width = 8


class Int8Narrow(Int8):
    narrow_range = True


class Uint8(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = False
    signed = False
    bit_width = 8


class FloatBias(Injector):
    quant_type = QuantType.FP
    narrow_range = False
    signed = True


class Int8BiasInternalFloatScaling(Injector):
    quant_type = QuantType.INT
    narrow_range = False
    signed = True
    bit_width = 8
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP


class TruncTo8bit(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.FLOOR
    bit_width = 8


Int8WeightPerTensorPoT = Int8Narrow & PerTensorPoTScaling & StatsMaxScaling
Int8WeightPerTensorFloat = Int8Narrow & PerTensorFloatScaling & StatsMaxScaling
Int8WeightPerChannelPoT = Int8Narrow & PerChannelPoTScaling & StatsMaxScaling
Int8WeightPerChannelFloat = Int8Narrow & PerChannelFloatScaling & StatsMaxScaling

Int8ActPerTensorPoT = Int8 & PerTensorPoTScaling & ParamFromRuntimePercentileScaling
Int8ActPerTensorFloat = Int8 & PerTensorFloatScaling & ParamFromRuntimePercentileScaling
Int8ActPerChannelPoT = Int8 & PerChannelPoTScaling & ParamFromRuntimePercentileScaling
Int8ActPerChannelFloat = Int8 & PerChannelFloatScaling & ParamFromRuntimePercentileScaling

Uint8ActPerTensorPoT = Uint8 & PerTensorPoTScaling & ParamFromRuntimePercentileScaling
Uint8ActPerTensorFloat = Uint8 & PerTensorFloatScaling & ParamFromRuntimePercentileScaling
Uint8ActPerChannelPoT = Uint8 & PerChannelPoTScaling & ParamFromRuntimePercentileScaling
Uint8ActPerChannelFloat = Uint8 & PerChannelFloatScaling & ParamFromRuntimePercentileScaling
