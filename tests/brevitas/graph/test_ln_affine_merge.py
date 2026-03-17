# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest
import torch
import torch.nn as nn

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import MergeLnAffine
from tests.marker import requires_pt_ge

ATOL = 1e-5


class LinearLnModel(nn.Module):

    def __init__(self, dtype=torch.float32, bias=True):
        super().__init__()
        self.ln = nn.LayerNorm(4, dtype=dtype)
        self.linear = nn.Linear(4, 3, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.linear(self.ln(x))


class LinearRmsModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.rms = nn.RMSNorm(4)
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(self.rms(x))


class MismatchedLinearLnModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(3)
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(self.ln(x[..., :3]))


def _apply_merge(model):
    graph_model = symbolic_trace(model.eval())
    return MergeLnAffine().apply(graph_model)


def _set_nontrivial_affine(module):
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.5, 0.5, 2.0, 0.25], dtype=module.weight.dtype))
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=module.bias.dtype))


def _merge_output_is_invariant(model, inp, atol=ATOL):
    model.eval()
    original_weight = model.linear.weight.detach().clone()
    with torch.no_grad():
        expected = model(inp)
    graph_model = _apply_merge(copy.deepcopy(model))
    with torch.no_grad():
        out = graph_model(inp)
    assert torch.allclose(out, expected, atol=atol)
    assert not torch.allclose(graph_model.linear.weight, original_weight)


def test_linear_ln_affine_merge_output_invariant():
    model = LinearLnModel()
    _set_nontrivial_affine(model.ln)
    inp = torch.randn(2, 4)
    _merge_output_is_invariant(model, inp)


@requires_pt_ge('2.4')
def test_linear_rms_affine_merge_output_invariant():
    model = LinearRmsModel()
    _set_nontrivial_affine(model.rms)
    inp = torch.randn(2, 4)
    _merge_output_is_invariant(model, inp)


def test_merge_keeps_low_precision_sink_dtype_and_uses_bias_registration():
    model = LinearLnModel(dtype=torch.float16, bias=False)
    _set_nontrivial_affine(model.ln)
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(
                [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, 2.0], [-1.0, 0.25, 0.75, -0.125]],
                dtype=torch.float16))

    expected_weight = (
        model.linear.weight.detach().to(torch.float64) *
        model.ln.weight.detach().to(torch.float64).view(1, -1)).to(torch.float16)
    expected_bias = torch.mv(
        expected_weight.detach().to(torch.float64),
        (model.ln.bias.detach().to(torch.float64) / model.ln.weight.detach().to(torch.float64))).to(
            torch.float16)

    graph_model = _apply_merge(copy.deepcopy(model))

    assert graph_model.linear.weight.dtype == torch.float16
    assert graph_model.linear.bias.dtype == torch.float16
    assert torch.equal(graph_model.linear.weight, expected_weight)
    assert torch.equal(graph_model.linear.bias, expected_bias)
