import brevitas.nn as quant_nn
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
import torch.nn as nn

QUANT_TYPE = QuantType.INT
QUANT_TYPE_BIAS = QuantType.FP

SCALING_MIN_VAL = 2e-9
ACT_SCALING_IMPL_TYPE = ScalingImplType.CONST
ACT_SCALING_PER_CHANNEL = False
ACT_MAX_VAL = 1
ACT_MIN_VAL = -1

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER_FROM_STATS
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_NARROW_RANGE = True
BIAS_CONFIGS = False


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_quantconv1d(feat_in, feat_out, kernel_size, stride, padding, bit_width, dilation=1, group=1):
    return quant_nn.QuantConv1d(in_channels=feat_in, out_channels=feat_out, kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=group,
                                weight_bit_width=bit_width,
                                weight_quant_type=QUANT_TYPE,
                                weight_narrow_range=WEIGHT_NARROW_RANGE,
                                weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                                weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                                weight_scaling_min_val=SCALING_MIN_VAL,
                                bias_bit_width=bit_width,
                                bias_quant_type=QUANT_TYPE_BIAS,
                                bias_narrow_range=BIAS_CONFIGS,
                                compute_output_scale=BIAS_CONFIGS,
                                compute_output_bit_width=BIAS_CONFIGS,
                                return_quant_tensor=False)


def make_transpconv1d(feat_in, feat_out, kernel_size, stride, padding, bit_width, dilation=1):
    return quant_nn.QuantConvTranspose1d(in_channels=feat_in, out_channels=feat_out, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         weight_bit_width=bit_width,
                                         weight_quant_type=QUANT_TYPE,
                                         weight_narrow_range=WEIGHT_NARROW_RANGE,
                                         weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                                         weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                                         weight_scaling_min_val=SCALING_MIN_VAL,
                                         bias_bit_width=bit_width,
                                         bias_quant_type=QUANT_TYPE_BIAS,
                                         bias_narrow_range=BIAS_CONFIGS,
                                         compute_output_scale=BIAS_CONFIGS,
                                         compute_output_bit_width=BIAS_CONFIGS,
                                         return_quant_tensor=False)


def make_relu_activation(bit_width):
    return quant_nn.QuantReLU(bit_width=bit_width,
                              max_val=ACT_MAX_VAL,
                              quant_type=QUANT_TYPE,
                              scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                              scaling_min_val=SCALING_MIN_VAL,
                              return_quant_tensor=False
                              )


def make_hardtanh_activation(bit_width, return_quant_tensor=False):
    return quant_nn.QuantHardTanh(bit_width=bit_width,
                                  max_val=ACT_MAX_VAL,
                                  min_val=ACT_MIN_VAL,
                                  quant_type=QUANT_TYPE,
                                  scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                                  scaling_min_val=SCALING_MIN_VAL,
                                  return_quant_tensor=return_quant_tensor
                                  )


def make_tanh_activation(bit_width):
    return quant_nn.QuantTanh(bit_width=bit_width,
                              quant_type=QUANT_TYPE,
                              scaling_min_val=SCALING_MIN_VAL,
                              return_quant_tensor=False
                              )


def make_leakyRelu_activation(bit_width):
    el1 = nn.LeakyReLU()
    el2 = make_hardtanh_activation(bit_width=bit_width)
    layer = nn.Sequential(el1, el2)

    return layer
