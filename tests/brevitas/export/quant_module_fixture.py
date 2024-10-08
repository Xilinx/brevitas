# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pytest_cases import fixture
from pytest_cases import parametrize
from pytest_cases import set_case_id
import torch
from torch import nn

from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantConv3d
from brevitas.nn import QuantConvTranspose1d
from brevitas.nn import QuantConvTranspose2d
from brevitas.nn import QuantConvTranspose3d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn import TruncAvgPool2d
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.scaled_int import Int8AccumulatorAwareWeightQuant
from brevitas.quant.scaled_int import Int8AccumulatorAwareZeroCenterWeightQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8BiasPerTensorFloatInternalScaling
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

from ...conftest import SEED

OUT_CH = 16
IN_CH = 8
IN_MEAN = 5
IN_SCALE = 3
FEATURES = 5
KERNEL_SIZE = 3
TOLERANCE = 1


class Int8AccumulatorawareZeroCenterWeightQuantPerTensorFloat(
        Int8AccumulatorAwareZeroCenterWeightQuant):
    scaling_per_output_type = ScalingPerOutputType.TENSOR


A2Q_QUANTIZERS = {
    'a2q_per_channel_float': (Int8AccumulatorAwareWeightQuant, Int8ActPerTensorFloat),
    'a2q_plus_per_tensor_float':
        (Int8AccumulatorawareZeroCenterWeightQuantPerTensorFloat, Int8ActPerTensorFloat)}

QUANTIZERS = {
    'asymmetric_per_tensor_float':
        (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_tensor_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'asymmetric_per_channel_float':
        (ShiftedUint8WeightPerChannelFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_channel_float': (Int8WeightPerChannelFloat, Int8ActPerTensorFloat),
    'symmetric_per_tensor_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint),
    'symmetric_per_channel_fixed_point':
        (Int8WeightPerChannelFixedPoint, Int8ActPerTensorFixedPoint),
    **A2Q_QUANTIZERS}

BIAS_QUANTIZERS = {
    'bias_external_scale': (Int32Bias,),
    'bias_internal_scale': (Int8BiasPerTensorFloatInternalScaling,)}

QUANT_WBIOL_IMPL = [
    QuantLinear,
    QuantConv1d,
    QuantConv2d,
    QuantConv3d,
    QuantConvTranspose1d,
    QuantConvTranspose2d,
    QuantConvTranspose3d,]

BIT_WIDTHS = [4, 8, 10]  # below 8, equal 8, above 8
BIAS_BIT_WIDTHS = [8, 16, 32]


@fixture
@parametrize('impl', QUANT_WBIOL_IMPL, ids=[f'{c.__name__}' for c in QUANT_WBIOL_IMPL])
def quant_module_impl(impl):
    return impl


@fixture
@parametrize('bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
def input_bit_width(bit_width):
    return bit_width


@fixture
@parametrize('bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
def weight_bit_width(bit_width):
    return bit_width


@fixture
@parametrize('bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
def output_bit_width(bit_width):
    return bit_width


@fixture
@parametrize('bit_width', BIAS_BIT_WIDTHS, ids=[f'b{b}' for b in BIAS_BIT_WIDTHS])
def bias_bit_width(bit_width):
    return bit_width


@fixture
@parametrize('quantizers', QUANTIZERS.items(), ids=list(QUANTIZERS.keys()))
def weight_act_quantizers(quantizers):
    return quantizers


@fixture
@parametrize('quantizers', A2Q_QUANTIZERS.items(), ids=list(A2Q_QUANTIZERS.keys()))
def a2q_weight_act_quantizers(quantizers):
    return quantizers


@fixture
@parametrize('quantizer', BIAS_QUANTIZERS.items(), ids=list(BIAS_QUANTIZERS.keys()))
def bias_quantizer(quantizer):
    return quantizer


@fixture
def quant_module(
        quant_module_impl,
        weight_act_quantizers,
        input_bit_width,
        weight_bit_width,
        output_bit_width,
        bias_bit_width,
        bias_quantizer):

    weight_act_quantizers_name, (weight_quant, io_quant) = weight_act_quantizers
    bias_quantizer_name, (bias_quant,) = bias_quantizer  # pytest needs an iterable

    if quant_module_impl == QuantLinear:
        layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
    else:
        layer_kwargs = {'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.layer = quant_module_impl(
                **layer_kwargs,
                bias=True,
                weight_quant=weight_quant,
                input_quant=io_quant,
                output_quant=io_quant,
                weight_bit_width=weight_bit_width,
                input_bit_width=input_bit_width,
                output_bit_width=output_bit_width,
                bias_bit_width=bias_bit_width,
                bias_quant=bias_quant,
                return_quant_tensor=True)

        def forward(self, x):
            return self.layer(x)

    torch.random.manual_seed(SEED)
    module = Model()
    yield module
    del module
