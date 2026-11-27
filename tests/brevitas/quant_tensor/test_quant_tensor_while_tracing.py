# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
QuantTensor operations while torch.compile or torch.export trace them.

Tensor values are unknown while tracing, so the operations that validate them skip their checks
with a warning, and integer F.linear returns a Tensor. The first tests emulate tracing in eager
mode by making is_fake_mode_active return True, the last ones compile with AOTAutograd.
"""

import pytest
import torch
import torch.nn.functional as F

from brevitas.quant_tensor import IntQuantTensor
from tests.marker import requires_torch_compile


def _int_qt(value=None, scale=0.5, zero_point=0.):
    value = torch.randn(2, 4) if value is None else value
    return IntQuantTensor(
        value, torch.tensor(scale), torch.tensor(zero_point), torch.tensor(8.), True, False)


def _trace(monkeypatch):
    for module in ('base_quant_tensor', 'int_torch_handler'):
        monkeypatch.setattr(f'brevitas.quant_tensor.{module}.is_fake_mode_active', lambda: True)


@pytest.mark.parametrize(
    'op, other_kwargs, error',
    [
        (lambda a, b: torch.cat([a, b]), dict(scale=0.25), 'Scaling factors are different'),
        (lambda a, b: torch.cat([a, b]), dict(zero_point=1.), 'Zero points are different'),
        (lambda a, b: a + b, dict(scale=0.25), 'Scaling factors are different'),
        (lambda a, b: a * b, dict(zero_point=1.), 'Zero-points of mul operands are non-zero'),
        (lambda a, b: a / b, dict(zero_point=1.), 'Zero-points of div operands are non-zero'),],
    ids=['cat-scale', 'cat-zero_point', 'add-scale', 'mul-zero_point', 'div-zero_point'])
def test_checks_are_skipped_while_tracing(monkeypatch, op, other_kwargs, error):
    a, b = _int_qt(), _int_qt(**other_kwargs)
    with pytest.raises(RuntimeError, match=error):
        op(a, b)  # eager validates the operands
    _trace(monkeypatch)
    with pytest.warns(UserWarning, match='Cannot check'):
        out = op(a, b)  # tracing assumes that the checks hold
    assert isinstance(out, IntQuantTensor)


def test_linear_returns_tensor_while_tracing(monkeypatch):
    x, w = _int_qt(torch.randn(2, 4)), _int_qt(torch.randn(3, 4))
    assert isinstance(F.linear(x, w), IntQuantTensor)  # eager returns the accumulator QuantTensor
    _trace(monkeypatch)
    with pytest.warns(UserWarning, match='Cannot check that the zero points are zero'):
        out = F.linear(x, w)  # the zero points cannot be read while tracing
    assert type(out) is torch.Tensor


# The QuantTensors are created inside the compiled function: passing one as an input fails
def _linear(x, w, scale, zero_point, bit_width):
    qt = lambda value: IntQuantTensor(value, scale, zero_point, bit_width, True, False)
    return F.linear(qt(x), qt(w))


def _cat(x, y, scale, other_scale, zero_point, bit_width):
    return torch.cat([
        IntQuantTensor(x, scale, zero_point, bit_width, True, False),
        IntQuantTensor(y, other_scale, zero_point, bit_width, True, False)])


@requires_torch_compile()
def test_linear_under_compile():
    x, w = torch.randn(2, 4), torch.randn(3, 4)
    metadata = torch.tensor(0.5), torch.tensor(0.), torch.tensor(8.)
    expected = _linear(x, w, *metadata)
    out = torch.compile(_linear, backend='aot_eager', fullgraph=True)(x, w, *metadata)
    assert type(out) is torch.Tensor
    assert torch.allclose(out, expected.value)


@requires_torch_compile()
def test_cat_under_compile():
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    metadata = torch.tensor(0.5), torch.tensor(0.25), torch.tensor(0.), torch.tensor(8.)
    with pytest.raises(RuntimeError, match='Scaling factors are different'):
        _cat(x, y, *metadata)  # eager validates the scaling factors
    out = torch.compile(_cat, backend='aot_eager', fullgraph=True)(x, y, *metadata)
    assert isinstance(out, IntQuantTensor)  # tracing skips the check
    assert torch.equal(out.value, torch.cat([x, y]))
