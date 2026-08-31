# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import platform

from hypothesis import given
from hypothesis import reproduce_failure
from packaging import version
import pytest
import pytest_cases
import torch

from brevitas import torch_version
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.gptq import gptq_mode
import brevitas.nn as qnn
from brevitas.nn.mixin import WeightRegion
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import ShiftedUint8ActPerTensorFloat
from brevitas.quant import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas_examples.common.generative.quantize import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from tests.brevitas.hyp_helper import float_tensor_st
from tests.marker import jit_disabled_for_compile
from tests.marker import requires_pt_ge
from tests.marker import requires_torch_compile


class Fp8PerRow(FP8e4m3OCPDynamicActPerRowFloat):
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(*shape[:-1], 1)
    permute_dims = None
    stats_reduce_dim = 1


WEIGHT_QUANTIZERS = {
    'int8': Int8WeightPerTensorFloat,
    'uint8': ShiftedUint8WeightPerTensorFloat,
    'fp8': Fp8e4m3WeightPerTensorFloat,
    'mxint8': MXInt8Weight,
    'mxfloat8': MXFloat8e4m3Weight}

ACT_QUANTIZERS = {
    'int8': Int8ActPerTensorFloat,
    'uint8': ShiftedUint8ActPerTensorFloat,
    'fp8': Fp8e4m3ActPerTensorFloat,
    'per_tensor_dynamic_int8': Int8DynamicActPerTensorFloat,
    'per_row_dynamic_fp8': Fp8PerRow,
    'mxint8': MXInt8Act,
    'mxfloat8': MXFloat8e4m3Act}


@pytest_cases.parametrize('weight_quantizer', WEIGHT_QUANTIZERS.items())
@given(weight=float_tensor_st(shape=(8, 16), max_val=1e10, min_val=-1e10))
@requires_pt_ge('2.3.1')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_weight(weight, weight_quantizer):
    name, quant = weight_quantizer
    if version.parse('2.8') <= torch_version < version.parse('2.9'):
        pytest.skip('Skipping due to random failures on torch 2.8.x')
    if name == 'mxfloat8' and torch_version == version.parse('2.3.1'):
        pytest.skip("Skip test for unknown failure. It works with more recent version of torch.")
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
    if torch_version >= version.parse('2.5.0') and torch_version < version.parse('2.8.0'):
        pytest.skip("Unknown compile error on torch versions above 2.5")
    inp = torch.randn(8, 16)
    linear = qnn.QuantLinear(16, 8, weight_quant=quant)
    linear.weight.data = weight
    linear.eval()
    out = linear.quant_weight().value

    linear.weight_quant.compile_quant()
    quant_out = linear.quant_weight().value
    with quant_inference_mode(linear, compile=True):
        _ = linear(inp)
        inference_out = linear.quant_weight()
    assert torch.allclose(out, quant_out)
    assert torch.allclose(out, inference_out)


@pytest.mark.parametrize(
    'weight_quantizer',
    [
        MXInt8Weight,
        MXFloat8e4m3Weight,
        IntWeightSymmetricGroupQuant.let(group_size=32),
        Fp8e4m3WeightSymmetricGroupQuant.let(group_size=32),
        ShiftedUint8WeightGroupQuantFloat.let(group_size=32)])
@requires_pt_ge('2.3.1')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_groupwise_weight_region(monkeypatch, weight_quantizer):
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
    if version.parse('2.5.0') <= torch_version < version.parse('2.8.0'):
        pytest.skip("Unknown compile error on torch versions above 2.5")

    linear = qnn.QuantLinear(33, 8, bias=False, weight_quant=weight_quantizer)
    linear.eval()
    linear(torch.randn(2, 33))
    expected = linear.quant_weight().value

    linear.weight_quant.compile_quant()

    def fail_full_quantization(*args, **kwargs):
        raise AssertionError("Compiled region request fell back to full-weight quantization")

    monkeypatch.setattr(linear, 'quant_weight', fail_full_quantization)
    assert linear.weight_quant.supports_quant_weight_region
    assert linear.weight_quant.is_region_quant_compiled
    for index in (0, 32, 0):
        actual = linear.quant_weight_region(WeightRegion((None, (index, index + 1))))
        assert torch.allclose(expected[:, index:index + 1], actual)


@requires_pt_ge('2.3.1')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_groupwise_parameter_from_stats_region(monkeypatch):
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
    if version.parse('2.5.0') <= torch_version < version.parse('2.8.0'):
        pytest.skip("Unknown compile error on torch versions above 2.5")

    quantizer = ShiftedUint8WeightGroupQuantFloat.let(
        group_size=32,
        scaling_impl_type='parameter_from_stats',
        zero_point_impl=ParameterFromStatsFromParameterZeroPoint)
    linear = qnn.QuantLinear(33, 8, bias=False, weight_quant=quantizer)
    linear.eval()
    linear(torch.randn(2, 33))
    expected = linear.quant_weight().value

    linear.weight_quant.compile_quant()

    def fail_full_quantization(*args, **kwargs):
        raise AssertionError("Compiled stateful region fell back to full quantization")

    monkeypatch.setattr(linear, 'quant_weight', fail_full_quantization)
    assert linear.weight_quant.is_region_quant_compiled
    for index in (0, 32, 0):
        actual = linear.quant_weight_region(WeightRegion((None, (index, index + 1))))
        assert torch.allclose(expected[:, index:index + 1], actual)


@pytest.mark.parametrize(
    'weight_quantizer',
    [
        MXInt8Weight,
        MXFloat8e4m3Weight,
        IntWeightSymmetricGroupQuant.let(group_size=32),
        Fp8e4m3WeightSymmetricGroupQuant.let(group_size=32)])
@requires_pt_ge('2.3.1')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_gptq_with_compiled_groupwise_weight_region(monkeypatch, weight_quantizer):
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
    if version.parse('2.5.0') <= torch_version < version.parse('2.8.0'):
        pytest.skip("Unknown compile error on torch versions above 2.5")

    model = qnn.QuantLinear(33, 8, bias=False, weight_quant=weight_quantizer)
    model.eval()
    inp = torch.randn(4, 33)
    model(inp)
    model.weight_quant.compile_quant()

    with gptq_mode(model, act_order=True, num_blocks=4) as gptq:
        gptq.model(inp)

        def fail_full_quantization(*args, **kwargs):
            raise AssertionError("Compiled GPTQ update fell back to full-weight quantization")

        monkeypatch.setattr(model, 'quant_weight', fail_full_quantization)
        gptq.update()

    assert model.weight_quant.is_region_quant_compiled
    assert torch.isfinite(model.weight).all()


@pytest_cases.parametrize('act_quantizer', ACT_QUANTIZERS.items())
@given(inp=float_tensor_st(shape=(8, 16), max_val=1e10, min_val=-1e10))
@requires_pt_ge('2.3.1')
@requires_torch_compile()
@jit_disabled_for_compile()
def test_compile_act(inp, act_quantizer):
    name, quant = act_quantizer
    if version.parse('2.8') <= torch_version < version.parse('2.9'):
        pytest.skip('Skipping due to random failures on torch 2.8.x')
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
    if torch_version >= version.parse('2.5.0') and torch_version < version.parse('2.8.0'):
        pytest.skip("Unknown compile error on torch versions above 2.5")
    if 'mx' in name:
        extra_kwargs = {'group_dim': 1}
    else:
        extra_kwargs = {}
    identity = qnn.QuantIdentity(quant, **extra_kwargs)
    out = identity(inp)
    identity.eval()
    out = identity(inp)

    identity.act_quant.compile_quant()
    quant_out = identity(inp)
    with quant_inference_mode(identity, compile=True):
        _ = identity(inp)
        inference_out = identity(inp)
    assert torch.allclose(out, quant_out)
    assert torch.allclose(out, inference_out)
