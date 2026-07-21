# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.nn.target.finn import PWPolyFEager
from brevitas.nn.target.finn import PWPolyFGELU
from brevitas.nn.target.finn import PWPolyFSigmoid
from brevitas.nn.target.finn import PWPolyFSiLU
from brevitas.nn.target.finn import PWPolyFTanh
from brevitas.quant_tensor import IntQuantTensor

PWPOLYF_CASES = [
    (PWPolyFGELU, "gelu"),
    (PWPolyFSiLU, "silu"),
    (PWPolyFSigmoid, "sigmoid"),
    (PWPolyFTanh, "tanh"),]


@pytest.mark.parametrize("pwpolyf_cls,_", PWPOLYF_CASES)
def test_pwpolyf_activation_forward_shape(pwpolyf_cls, _):
    mod = pwpolyf_cls(K=2, degree=2)
    inp = torch.randn(2, 3, 4)

    out = mod(inp)

    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    assert torch.isfinite(out).all().item()


@pytest.mark.parametrize("pwpolyf_cls,func", PWPOLYF_CASES)
def test_pwpolyf_activation_approximates_reference(pwpolyf_cls, func):
    mod = pwpolyf_cls(K=3, degree=2)
    inp = torch.linspace(-4.0, 4.0, steps=65)

    out = mod(inp)
    ref = mod.eager_impl.act_impl(inp)

    assert mod.func == func
    assert torch.allclose(out, ref, atol=1e-1)


@pytest.mark.parametrize(
    "pwpolyf_cls,neg_expected,pos_expected", [
        (PWPolyFGELU, 0.0, 9.0),
        (PWPolyFSiLU, 0.0, 9.0),
        (PWPolyFSigmoid, 0.0, 1.0),
        (PWPolyFTanh, -1.0, 1.0),])
def test_pwpolyf_activation_clamps(pwpolyf_cls, neg_expected, pos_expected):
    mod = pwpolyf_cls(K=2, degree=2)
    inp = torch.tensor([-9.0, 9.0])

    out = mod(inp)

    assert torch.allclose(out, torch.tensor([neg_expected, pos_expected]))


def test_pwpolyf_activation_accepts_quant_tensor_input():
    mod = PWPolyFGELU(K=2, degree=2)
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


def test_pwpolyf_eager_is_registered_module():
    mod = PWPolyFGELU(K=2, degree=2)

    assert isinstance(mod.eager_impl, PWPolyFEager)
    assert dict(mod.named_children())["eager_impl"] is mod.eager_impl
    assert dict(mod.eager_impl.named_buffers())["coeffs"] is mod.coeffs


@pytest.mark.parametrize("kwargs", [{"K": 0}, {"degree": 0}])
def test_pwpolyf_activation_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PWPolyFGELU(**kwargs)


def test_pwpolyf_activation_rejects_non_float32_input():
    mod = PWPolyFGELU(K=2, degree=2)
    inp = torch.randn(8, dtype=torch.float64)

    with pytest.raises(ValueError, match="torch.float32"):
        mod(inp)
