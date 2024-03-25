# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union
import torch

from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantConv3d
from brevitas.nn import QuantConvTranspose1d
from brevitas.nn import QuantConvTranspose2d
from brevitas.nn import QuantConvTranspose3d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantScaleBias
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias
from brevitas.quant.scaled_int import Int8BiasPerTensorFloatInternalScaling
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from tests.brevitas.common import BOOLS

OUTPUT_CH = 10
IN_CH = 5
KERNEL_SIZE = 3

QUANT_CONV_VARIANTS = [
    QuantConv1d,
    QuantConv2d,
    QuantConv3d,
    QuantConvTranspose1d,
    QuantConvTranspose2d,
    QuantConvTranspose3d,]


@pytest_cases.fixture()
@pytest_cases.parametrize('is_enabled', BOOLS)
def bias_enabled(is_enabled):
    """
    Bias enabled in QuantWBIOL layer
    """
    return is_enabled


@pytest_cases.fixture()
def default_wbiol_quant_linear(bias_enabled):
    """
    QuantLinear layer with default quantization settings
    """
    return QuantLinear(out_features=OUTPUT_CH, in_features=IN_CH, bias=bias_enabled)


@pytest_cases.fixture()
@pytest_cases.parametrize('quant_conv_variant', QUANT_CONV_VARIANTS)
def default_wbiol_quant_conv(quant_conv_variant, bias_enabled):
    """
    QuantConv(Tranpose)(1/2)d layer with default quantization settings
    """
    return quant_conv_variant(
        out_channels=OUTPUT_CH, in_channels=IN_CH, kernel_size=KERNEL_SIZE, bias=bias_enabled)


@pytest_cases.fixture()
def default_weight_tensor_quant(default_wbiol_layer):
    """
    Returns default_wbiol_layer.weight_quant.tensor_quant
    """
    return default_wbiol_layer.weight_quant.tensor_quant


# Union of all WBIOL layers with default quantization settings
fixture_union('default_wbiol_layer', ['default_wbiol_quant_linear', 'default_wbiol_quant_conv'])


def test_default_wbiol_input_quant_enabled(default_wbiol_layer: QuantWBIOL):
    assert not default_wbiol_layer.input_quant.is_quant_enabled


def test_default_wbiol_output_quant_enabled(default_wbiol_layer: QuantWBIOL):
    assert not default_wbiol_layer.output_quant.is_quant_enabled


def test_default_wbiol_bias_quant_enabled(default_wbiol_layer: QuantWBIOL):
    assert not default_wbiol_layer.bias_quant.is_quant_enabled


def test_default_wbiol_weight_quant_enabled(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.weight_quant.is_quant_enabled


def test_default_wbiol_weight_bit_width_enabled(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.weight_quant.bit_width() == torch.tensor(8.)


def test_default_wbiol_return_quant(default_wbiol_layer: QuantWBIOL):
    assert not default_wbiol_layer.return_quant_tensor


def test_default_wbiol_quant_bias_signed(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.bias_quant.is_signed is None


def test_default_wbiol_quant_weight_signed(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.weight_quant.is_signed


def test_default_wbiol_quant_bias_narrow_range(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.bias_quant.is_narrow_range is None


def test_default_wbiol_quant_weight_narrow_range(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.weight_quant.is_narrow_range


def test_default_wbiol_quant_input_signed(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.input_quant.is_signed is None


def test_default_wbiol_quant_output_signed(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.output_quant.is_signed is None


def test_default_wbiol_quant_input_narrow_range(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.input_quant.is_narrow_range is None


def test_default_wbiol_quant_output_narrow_range(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.output_quant.is_narrow_range is None


def test_default_wbiol_quant_input_zero_point(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.input_quant.zero_point() is None


def test_default_wbiol_quant_output_zero_point(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.output_quant.zero_point() is None


def test_default_wbiol_quant_weight_zero_point(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.weight_quant.zero_point() == torch.tensor(0.)


def test_default_wbiol_quant_bias_zero_point(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.bias_quant.zero_point() is None


def test_default_wbiol_quant_input_scale(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.input_quant.scale() is None


def test_default_wbiol_quant_output_scale(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.output_quant.scale() is None


def test_default_wbiol_quant_bias_scale(default_wbiol_layer: QuantWBIOL):
    assert default_wbiol_layer.bias_quant.scale() is None


def test_default_wbiol_weight_quant_proxy(default_wbiol_layer: QuantWBIOL):
    assert isinstance(default_wbiol_layer.weight_quant, WeightQuantProxyFromInjector)


def test_default_wbiol_input_quant_proxy(default_wbiol_layer: QuantWBIOL):
    assert isinstance(default_wbiol_layer.input_quant, ActQuantProxyFromInjector)


def test_default_wbiol_output_quant_proxy(default_wbiol_layer: QuantWBIOL):
    assert isinstance(default_wbiol_layer.output_quant, ActQuantProxyFromInjector)


def test_default_wbiol_bias_quant_proxy(default_wbiol_layer: QuantWBIOL):
    assert isinstance(default_wbiol_layer.bias_quant, BiasQuantProxyFromInjector)


def test_default_wbiol_quant_injector(default_wbiol_layer: QuantWBIOL):
    assert issubclass(default_wbiol_layer.weight_quant.quant_injector, Int8WeightPerTensorFloat)


def test_internally_scaled_bias_zero_point():
    conv = QuantConv2d(
        IN_CH,
        OUTPUT_CH,
        KERNEL_SIZE,
        bias=True,
        input_quant=Int8ActPerTensorFloat,
        bias_quant=Int8BiasPerTensorFloatInternalScaling,
        return_quant_tensor=True)
    out = conv(torch.randn(1, IN_CH, 10, 10))
    assert (out.zero_point != 0.).any()


def test_float_bias_zero_point():
    conv = QuantConv2d(
        IN_CH,
        OUTPUT_CH,
        KERNEL_SIZE,
        bias=True,
        input_quant=Int8ActPerTensorFloat,
        return_quant_tensor=True)
    out = conv(torch.randn(1, IN_CH, 10, 10))
    assert (out.zero_point != 0.).any()


def test_externally_scaled_bias_zero_point():
    conv = QuantConv2d(
        IN_CH,
        OUTPUT_CH,
        KERNEL_SIZE,
        bias=True,
        input_quant=Int8ActPerTensorFloat,
        bias_quant=Int8Bias,
        return_quant_tensor=True)
    out = conv(torch.randn(1, IN_CH, 10, 10))
    assert (out.zero_point == 0.).any()
