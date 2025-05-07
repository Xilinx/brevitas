import platform

from hypothesis import given
from hypothesis import reproduce_failure
from packaging import version
import pytest
import pytest_cases
import torch

from brevitas import torch_version
from brevitas.export.inference import quant_inference_mode
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import ShiftedUint8ActPerTensorFloat
from brevitas.quant import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas_examples.common.generative.quantize import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from tests.brevitas.hyp_helper import float_tensor_st
from tests.marker import jit_disabled_for_compile
from tests.marker import requires_pt_ge


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
@jit_disabled_for_compile()
def test_compile_weight(weight, weight_quantizer):
    name, quant = weight_quantizer
    if name == 'mxfloat8' and torch_version == version.parse('2.3.1'):
        pytest.skip("Skip test for unknown failure. It works with more recent version of torch.")
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
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


@pytest_cases.parametrize('act_quantizer', ACT_QUANTIZERS.items())
@given(inp=float_tensor_st(shape=(8, 16), max_val=1e10, min_val=-1e10))
@requires_pt_ge('2.3.1')
@jit_disabled_for_compile()
def test_compile_act(inp, act_quantizer):
    name, quant = act_quantizer
    if platform.system() == "Windows":
        pytest.skip("Skip compile + windows because of unknown failure")
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
