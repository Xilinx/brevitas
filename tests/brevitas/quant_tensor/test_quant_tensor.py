# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum

import numpy as np
from packaging import version
import pytest
import pytest_cases
import torch

from brevitas import torch_version
from brevitas.nn import QuantIdentity
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPActPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIO
from brevitas.utils.quant_utils import _CachedIOFloat
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class Operator(Enum):
    ADD = 0
    SUBTRACT = 1
    DIVIDE = 2
    MULTIPLY = 3
    MATMUL = 4


def to_quant_tensor(input: torch.Tensor, bit_width=8) -> IntQuantTensor:
    mod = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
    return mod(input)


def to_float_quant_tensor(
        input: torch.Tensor,
        bit_width=8,
        exponent_bit_width=4,
        mantissa_bit_width=3) -> FloatQuantTensor:
    mod = QuantIdentity(
        bit_width=bit_width,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        return_quant_tensor=True,
        act_quant=Fp8e5m2OCPActPerTensorFloat)
    return mod(input)


def to_mx_quant_tensor(
        input: torch.Tensor,
        bit_width=8,
        exponent_bit_width=4,
        mantissa_bit_width=3,
        group_size=32,
        group_dim=1) -> GroupwiseFloatQuantTensor:
    mod = QuantIdentity(
        bit_width=bit_width,
        group_size=group_size,
        group_dim=group_dim,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        return_quant_tensor=True,
        act_quant=MXFloat8e4m3Act)
    return mod(input)


def qdq(normal_tensor, quant_tensor):
    return (
        torch.round(normal_tensor / quant_tensor.scale + quant_tensor.zero_point) -
        quant_tensor.zero_point) * quant_tensor.scale


def test_qt_structure():
    qt = IntQuantTensor(
        torch.randn(10), torch.randn(1), torch.tensor(0.), torch.tensor(8.), True, False)
    assert isinstance(qt, IntQuantTensor)
    assert isinstance(qt, QuantTensor)
    assert isinstance(qt, tuple)
    assert hasattr(qt, '_fields')
    assert len(qt._fields) == 6


def test_quant_tensor_init():
    x = torch.randn(4, 4)
    quant_tensor = to_quant_tensor(x)
    normal_tensor = torch.Tensor(x)
    assert torch.allclose(qdq(normal_tensor, quant_tensor), quant_tensor, rtol=0.01)


@pytest.mark.parametrize(
    'op', [Operator.ADD, Operator.SUBTRACT, Operator.DIVIDE, Operator.MULTIPLY, Operator.MATMUL])
@pytest.mark.parametrize('quant_fn', [to_quant_tensor, to_float_quant_tensor])
def test_quant_tensor_operators(op, quant_fn):

    if quant_fn == to_float_quant_tensor and torch_version < version.parse('1.13'):
        pytest.skip("Torch 1.13 is required for JIT to be compatible with FloatQuantTensor")

    # Avoid 0 values
    x = 1 + torch.rand(4, 4)

    a = torch.Tensor(x)
    b = torch.Tensor(x)

    qa = quant_fn(a)
    qb = quant_fn(b)

    # to factor in quantisation error
    e_a = a - qa
    e_b = b - qb

    if op == Operator.ADD:
        quant = qa + qb
        normal = (a - e_a) + (b - e_b)
    elif op == Operator.SUBTRACT:
        quant = qa - qb
        normal = (a - e_a) - (b - e_b)
    elif op == Operator.DIVIDE:
        quant = qa / qb
        normal = (a - e_a) / (b - e_b)
    elif op == Operator.MULTIPLY:
        quant = qa * qb
        normal = (a - e_a) * (b - e_b)
    elif op == Operator.MATMUL:
        # @ matmul operator not implemented for QuantTensor
        quant = torch.matmul(qa, qb)
        normal = (a - e_a) @ (b - e_b)
    else:
        # unrecognised operator
        assert False

    assert torch.allclose(normal, quant)


def test_quant_tensor_div_by_zero():
    a = to_quant_tensor(torch.ones(4, 4))
    b = to_quant_tensor(torch.zeros(4, 4))
    assert torch.isinf(a / b).all().item()


def test_quant_tensor_div_by_fraction():
    a = to_quant_tensor(torch.ones(4, 4))
    b = to_quant_tensor(torch.ones(4, 4) * 0.5)
    assert torch.allclose(a / b, torch.ones(4, 4) * 2, atol=0.1)


# TODO: need to deal with quant metadata
def test_quant_tensor_transpose():
    x = torch.ones(4, 4).tril()
    a = x.clone()
    b = to_quant_tensor(x)
    assert torch.allclose(a.transpose(0, 1), b.transpose(0, 1), atol=0.01)


# TODO: need to deal with quant metadata
def test_quant_tensor_view():
    x = torch.ones(4, 4)
    a = to_quant_tensor(x)
    b = torch.Tensor(x)

    assert torch.allclose(a.view(-1), b.view(-1), atol=0.01)
    assert torch.allclose(a.view(2, -1), b.view(2, -1), atol=0.01)
    assert torch.allclose(a.view(16, -1), b.view(16, -1), atol=0.01)
    assert torch.allclose(a.view(8, 2), b.view(8, 2), atol=0.01)


QUANT_CLASS = {'fp8': Fp8e4m3ActPerTensorFloat, 'mxfp8': MXFloat8e4m3Act}


@pytest_cases.parametrize('quant_class_key_vale', QUANT_CLASS.items())
def test_minifloat(quant_class_key_vale):
    key, quant_class = quant_class_key_vale

    x = torch.randn((1, 32))
    q = QuantIdentity(quant_class, group_dim=-1, return_quant_tensor=True)
    q.eval()

    qx = q(x)
    # Check that minifloat doesn't raise error
    qx.minifloat()


@pytest.mark.parametrize("metadata_only", [True, False])
def test_int_quant_tensor(metadata_only, bit_width=8):
    limit = np.exp2(bit_width) - 1
    w = torch.randn(32, 1024)
    q = to_quant_tensor(w, bit_width=bit_width)
    i = q.int().float()
    assert ((i.max() - i.min()) <= limit).all()
    # test caching works
    cache = _CachedIO(q, metadata_only=metadata_only)
    assert cache.bit_width == bit_width


@pytest.mark.parametrize("metadata_only", [True, False])
def test_float_quant_tensor(metadata_only, bit_width=8, exponent_bit_width=4, mantissa_bit_width=3):
    assert mantissa_bit_width + exponent_bit_width + 1 == bit_width
    limit = (np.exp2(mantissa_bit_width + 1) - 1) * np.exp2(np.exp2(exponent_bit_width) - 2)
    w = torch.randn(32, 1024)
    q = to_float_quant_tensor(
        w,
        bit_width=bit_width,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width)
    # test that the integer API returns fixed point values in the right range
    i = q.int().float()
    assert ((i.max() - i.min()) <= limit).all()
    # test caching works
    cache = _CachedIOFloat(q, metadata_only=metadata_only)
    assert cache.mantissa_bit_width == mantissa_bit_width
    assert cache.exponent_bit_width == exponent_bit_width


@pytest.mark.parametrize("metadata_only", [True, False])
def test_mx_quant_tensor(metadata_only, bit_width=8, exponent_bit_width=4, mantissa_bit_width=3):
    assert mantissa_bit_width + exponent_bit_width + 1 == bit_width
    limit = (np.exp2(mantissa_bit_width + 1) - 1) * np.exp2(np.exp2(exponent_bit_width) - 2)
    w = torch.randn(32, 1024)
    q = to_mx_quant_tensor(
        w,
        bit_width=bit_width,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        group_size=32,
        group_dim=1)
    # test that the integer API returns fixed point values in the right range
    i = q.int().float()
    assert ((i.max() - i.min()) <= limit).all()
    # test caching works
    cache = _CachedIOGroupwiseFloat(q, metadata_only=metadata_only)
    assert cache.mantissa_bit_width == mantissa_bit_width
    assert cache.exponent_bit_width == exponent_bit_width
    assert cache.group_size == 32
    assert cache.group_dim == 1
