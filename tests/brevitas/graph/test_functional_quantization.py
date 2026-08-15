# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.graph.functional_quant import functional_quantization_mode
from brevitas.graph.functional_quant import prepare_functional_quantization
from brevitas.quant import Int8ActPerTensorFloat


class TwoLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x, second=True):
        x = self.linear1(x)
        return self.linear2(x) if second else x


class ReusedFunctionalModel(nn.Module):

    def forward(self, x):
        return F.linear(F.linear(x, torch.eye(4)), torch.eye(4))


def test_prepares_and_applies_quantizers():
    model = TwoLinearModel()
    x = torch.randn(2, 4)
    state = prepare_functional_quantization(model, {F.linear: Int8ActPerTensorFloat}, (x,))
    assert len(state.quantizers) == 2
    with functional_quantization_mode(state):
        assert model(x).shape == (2, 2)
    state.cleanup()


def test_counts_repeated_calls_in_one_module():
    model = ReusedFunctionalModel()
    x = torch.randn(2, 4)
    state = prepare_functional_quantization(model, {F.linear: Int8ActPerTensorFloat}, (x,))
    # Both ownerless runtime weights use activation fallback. The outer input is
    # already a QuantTensor from the inner call and is not quantized again.
    assert len(state.quantizers) == 3
    state.cleanup()


def test_unprepared_call_site_fails_fast():
    model = TwoLinearModel()
    x = torch.randn(2, 4)
    state = prepare_functional_quantization(
        model, {F.linear: Int8ActPerTensorFloat}, (x,), {'second': False})
    with functional_quantization_mode(state):
        try:
            model(x, second=True)
            assert False, 'Expected an unprepared call-site error.'
        except RuntimeError as error:
            assert 'No prepared quantizer' in str(error)
    state.cleanup()


def test_cleanup_removes_container():
    model = TwoLinearModel()
    state = prepare_functional_quantization(
        model, {F.linear: Int8ActPerTensorFloat}, (torch.randn(2, 4),))
    state.cleanup()
    assert not hasattr(model, '_functional_quantizers')


def test_missing_second_runtime_spec_reuses_first_quantizer():

    class BmmModel(nn.Module):

        def forward(self, left, right):
            return torch.bmm(left, right)

    state = prepare_functional_quantization(
        BmmModel(), {torch.bmm: Int8ActPerTensorFloat},
        (torch.randn(2, 3, 4), torch.randn(2, 4, 3)))
    assert len(state.quantizers) == 2
    state.cleanup()


def test_explicit_none_skips_argument():

    class BmmModel(nn.Module):

        def forward(self, left, right):
            return torch.bmm(left, right)

    state = prepare_functional_quantization(
        BmmModel(), {torch.bmm: (Int8ActPerTensorFloat, None)},
        (torch.randn(2, 3, 4), torch.randn(2, 4, 3)))
    assert len(state.quantizers) == 1
    state.cleanup()
