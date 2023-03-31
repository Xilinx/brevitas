# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
import brevitas.nn as quant_nn

global ACT_MIN_VAL, ACT_MAX_VAL
brevitas_activations = {
    "hardtanh": quant_nn.QuantHardTanh,
    "relu": quant_nn.QuantReLU,}

QUANT_TYPE = QuantType.INT
QUANT_TYPE_BIAS = QuantType.FP

SCALING_MIN_VAL = 2e-16
ACT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL = False
ACT_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.STATS
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_NARROW_RANGE = True
BIAS_CONFIGS = False


def make_quantization_input(bit_width, absolute_act_val, scaling_per_channel):
    return quant_nn.QuantHardTanh(
        bit_width=bit_width,
        scaling_per_output_channel=scaling_per_channel,
        quant_type=QUANT_TYPE,
        scaling_impl_type=ACT_SCALING_IMPL_TYPE,
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_type=ACT_RESTRICT_SCALING_TYPE,
        max_val=absolute_act_val,
        min_val=-absolute_act_val,
        return_quant_tensor=False)


def make_norm_scale(bit_width, absolute_act_val, scaling_per_channel):
    return quant_nn.QuantHardTanh(
        bit_width=bit_width,
        scaling_per_output_channel=scaling_per_channel,
        quant_type=QUANT_TYPE,
        scaling_impl_type=ACT_SCALING_IMPL_TYPE,
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_type=ACT_RESTRICT_SCALING_TYPE,
        max_val=absolute_act_val,
        min_val=-absolute_act_val,
        scaling_stats_permute_dims=(1, 0, 2),
        return_quant_tensor=True)


def make_jasper_activation(activation, channels, bit_width, absolute_act_val, scaling_per_channel):
    brevitas_activation = brevitas_activations[activation]
    return brevitas_activation(
        bit_width=bit_width,
        scaling_per_output_channel=scaling_per_channel,
        quant_type=QUANT_TYPE,
        scaling_impl_type=ACT_SCALING_IMPL_TYPE,
        scaling_min_val=SCALING_MIN_VAL,
        restrict_scaling_type=ACT_RESTRICT_SCALING_TYPE,
        max_val=absolute_act_val,
        per_channel_broadcastable_shape=(1, channels, 1),
        scaling_stats_permute_dims=(1, 0, 2),
        return_quant_tensor=False)


def make_quantconv1d(
        feat_in,
        classes,
        kernel_size,
        bit_width,
        scaling_per_channel,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1):

    return quant_nn.QuantConv1d(
        in_channels=feat_in,
        out_channels=classes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        weight_bit_width=bit_width,
        weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
        weight_scaling_per_output_channel=scaling_per_channel,
        weight_quant_type=QUANT_TYPE,
        weight_narrow_range=WEIGHT_NARROW_RANGE,
        weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
        weight_scaling_min_val=SCALING_MIN_VAL,
        bias_bit_width=bit_width,
        bias_quant_type=QUANT_TYPE_BIAS,
        bias_narrow_range=BIAS_CONFIGS,
        compute_output_scale=BIAS_CONFIGS,
        compute_output_bit_width=BIAS_CONFIGS,
        return_quant_tensor=False)
