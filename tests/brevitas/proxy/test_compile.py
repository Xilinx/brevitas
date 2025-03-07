import pytest_cases
import torch

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
    'mxint8': MXInt8Act,
    'mxfloat8': MXFloat8e4m3Act}


@pytest_cases.parametrize('weight_quantizer', WEIGHT_QUANTIZERS.items())
def test_compile(weight_quantizer):
    name, quant = weight_quantizer
    linear = qnn.QuantLinear(8, 128, weight_quant=quant)
    out = linear.quant_weight()
    linear.weight_quant.compile_quant()
    quant_out = linear.quant_weight()
    assert torch.allclose(out, quant_out)


@pytest_cases.parametrize('act_quantizer', ACT_QUANTIZERS.items())
def test_compile(act_quantizer):
    name, quant = act_quantizer
    if 'mx' in name:
        extra_kwargs = {'group_dim': 1}
    else:
        extra_kwargs = {}
    inp = torch.randn(8, 128)
    identity = qnn.QuantIdentity(quant, **extra_kwargs)
    out = identity(inp)
    identity.eval()
    out = identity(inp)

    identity.act_quant.compile_quant()
    quant_out = identity(inp)
    assert torch.allclose(out, quant_out)
