# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.nn import QuantCat
from brevitas.nn import QuantEltwiseAdd
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor

# Int8ActPerTensorFloatMinMaxInit uses the fixed scale max(|min_val|, |max_val|) / 128, so the
# expected outputs can be computed by hand whatever the inputs and the train/eval mode are.
# Power-of-two ranges keep the scales exact in floating point.
INPUT_MAX_VAL = 4.0
OUTPUT_MAX_VAL = 8.0
FIXED_SCALE_KWARGS = dict(
    input_quant=Int8ActPerTensorFloatMinMaxInit,
    input_min_val=-INPUT_MAX_VAL,
    input_max_val=INPUT_MAX_VAL,
    output_quant=Int8ActPerTensorFloatMinMaxInit,
    output_min_val=-OUTPUT_MAX_VAL,
    output_max_val=OUTPUT_MAX_VAL,
    return_quant_tensor=True)


def fake_quant(x, max_val):
    scale = max_val / 128
    return torch.clamp(torch.round(x / scale), -128, 127) * scale


class TestQuantEltwiseAdd:

    def test_forward_defaults(self):
        mod = QuantEltwiseAdd()
        out = mod(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4))
        assert out.shape == (2, 3, 4, 4)
        assert not isinstance(out, QuantTensor)

    def test_forward_fixed_scales(self):
        mod = QuantEltwiseAdd(**FIXED_SCALE_KWARGS).eval()
        a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        out = mod(a, b)

        qa, qb = fake_quant(a, INPUT_MAX_VAL), fake_quant(b, INPUT_MAX_VAL)
        expected = fake_quant(qa + qb, OUTPUT_MAX_VAL)
        assert isinstance(out, IntQuantTensor)
        assert out.is_valid
        assert out.scale == OUTPUT_MAX_VAL / 128
        assert out.bit_width == 8
        torch.testing.assert_close(out.value, expected)


class TestQuantCat:

    def test_forward_defaults(self):
        mod = QuantCat()
        out = mod([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
        assert out.shape == (2, 6, 4)  # dim=1 by default
        assert not isinstance(out, QuantTensor)

    @pytest.mark.parametrize('dim', [0, 1, -1])
    def test_forward_fixed_scales(self, dim):
        mod = QuantCat(**FIXED_SCALE_KWARGS).eval()
        a, b = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        out = mod([a, b], dim=dim)

        qa, qb = fake_quant(a, INPUT_MAX_VAL), fake_quant(b, INPUT_MAX_VAL)
        expected = fake_quant(torch.cat([qa, qb], dim=dim), OUTPUT_MAX_VAL)
        assert isinstance(out, IntQuantTensor)
        assert out.is_valid
        assert out.scale == OUTPUT_MAX_VAL / 128
        assert out.bit_width == 8
        torch.testing.assert_close(out.value, expected)
