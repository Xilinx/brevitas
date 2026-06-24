# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.nn.target.finn import PWPolyFEager
from brevitas.nn.target.finn import PWPolyFActivation
from brevitas.quant_tensor import IntQuantTensor


@pytest.mark.parametrize("func", PWPolyFEager.supported_funcs())
def test_pwpolyf_activation_forward_shape(func):
    mod = PWPolyFActivation(func=func, K=2, degree=2)
    inp = torch.randn(2, 3, 4)

    out = mod(inp)

    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    assert torch.isfinite(out).all().item()


@pytest.mark.parametrize("func", PWPolyFEager.supported_funcs())
def test_pwpolyf_activation_approximates_reference(func):
    mod = PWPolyFActivation(func=func, K=3, degree=2)
    inp = torch.linspace(-4.0, 4.0, steps=65)

    out = mod(inp)
    ref = mod.eager_impl.function_spec.reference_impl(inp)

    assert torch.allclose(out, ref, atol=1e-1)


@pytest.mark.parametrize(
    "func,neg_expected,pos_expected", [
        ("gelu", 0.0, 9.0),
        ("silu", 0.0, 9.0),
        ("sigmoid", 0.0, 1.0),
        ("tanh", -1.0, 1.0),])
def test_pwpolyf_activation_clamps(func, neg_expected, pos_expected):
    mod = PWPolyFActivation(func=func, K=2, degree=2)
    inp = torch.tensor([-9.0, 9.0])

    out = mod(inp)

    assert torch.allclose(out, torch.tensor([neg_expected, pos_expected]))


def test_pwpolyf_activation_accepts_quant_tensor_input():
    mod = PWPolyFActivation(func="gelu", K=2, degree=2)
    inp = torch.randn(1, 8)
    quant_inp = IntQuantTensor(
        value=inp,
        scale=torch.tensor(0.1),
        zero_point=torch.tensor(0.0),
        bit_width=torch.tensor(8.0),
        signed=torch.tensor(True),
        training=torch.tensor(False))

    out = mod(quant_inp)

    assert isinstance(out, torch.Tensor)
    assert out.shape == inp.shape


def test_pwpolyf_activation_rejects_unknown_func():
    with pytest.raises(ValueError, match="Unsupported func"):
        PWPolyFActivation(func="relu")


@pytest.mark.parametrize("kwargs", [{"K": 0}, {"degree": 0}])
def test_pwpolyf_activation_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PWPolyFActivation(**kwargs)


def test_pwpolyf_activation_rejects_non_float32_input():
    mod = PWPolyFActivation(func="gelu", K=2, degree=2)
    inp = torch.randn(8, dtype=torch.float64)

    with pytest.raises(ValueError, match="torch.float32"):
        mod(inp)
