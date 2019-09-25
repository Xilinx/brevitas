import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp

QUANT_TYPE = QuantType.INT
SCALING_MIN_VAL = 2e-32

ACT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL = False
ACT_SCALING_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
ACT_MAX_VAL = 6.0

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.AFFINE_STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE = True


def make_quant_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, bit_width):
    return qnn.QuantConv2d(in_channels, out_channels,
                           kernel_size=kernel_size,
                           padding=padding,
                           stride=stride,
                           bias=bias,
                           weight_bit_width=bit_width,
                           weight_quant_type=QUANT_TYPE,
                           weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                           weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                           weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                           weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                           weight_narrow_range=WEIGHT_NARROW_RANGE,
                           weight_scaling_min_val=SCALING_MIN_VAL)


def make_quant_linear(in_channels, out_channels, bias, bit_width,
                      scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL):
    return qnn.QuantLinear(in_channels, out_channels,
                           bias=bias,
                           weight_bit_width=bit_width,
                           weight_quant_type=QUANT_TYPE,
                           weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                           weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                           weight_scaling_per_output_channel=scaling_per_output_channel,
                           weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                           weight_narrow_range=WEIGHT_NARROW_RANGE,
                           weight_scaling_min_val=SCALING_MIN_VAL)


def make_quant_relu(bit_width):
    return qnn.QuantReLU(bit_width=bit_width,
                         quant_type=QUANT_TYPE,
                         scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                         scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                         restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                         scaling_min_val=SCALING_MIN_VAL,
                         max_val=ACT_MAX_VAL)

