# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import torch

from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
from tests.brevitas.hyp_helper import float_tensor_random_shape_st

from .minifloat_fixtures import *

FORMATS = {Fp8e5m2Mixin: 57344., Fp8e4m3Mixin: 448.}


@pytest.mark.parametrize(
    'minifloat, expected_max_val', ((format, max_val) for format, max_val in FORMATS.items()))
def test_max_value(minifloat, expected_max_val):
    # minifloat_format, expected_max_val = format
    exponent_bit_width = torch.tensor(minifloat.exponent_bit_width)
    mantissa_bit_width = torch.tensor(minifloat.mantissa_bit_width)
    exponent_bias = torch.tensor(minifloat.exponent_bias)

    max_val = minifloat.case_clamp_impl.get_max_value(
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        exponent_bias=exponent_bias)

    assert expected_max_val == max_val


@given(inp=float_tensor_random_shape_st())
def test_clamp(inp, fp8_clamp):
    # construct tensor which exceeds max val
    exponent_bit_width = torch.tensor(fp8_clamp.exponent_bit_width)
    mantissa_bit_width = torch.tensor(fp8_clamp.mantissa_bit_width)
    exponent_bias = torch.tensor(fp8_clamp.exponent_bias)

    max_val = fp8_clamp.case_clamp_impl.get_max_value(
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        exponent_bias=exponent_bias)
    # get values that exceed max_val
    over_limit_mask = inp.abs() > max_val

    # clamp inp
    inp = fp8_clamp.case_clamp_impl(inp, exponent_bit_width, mantissa_bit_width, exponent_bias)

    if fp8_clamp.case_clamp_impl.saturating:
        # should be clamped to +- max val
        assert (inp[over_limit_mask].abs() == max_val).all()
    else:
        # if inf_values, over limit mask should now be all inf
        if len(fp8_clamp.case_clamp_impl.inf_values) > 0:
            # all values exceeding max_val should be inf
            assert inp[over_limit_mask].isinf().all()
        else:
            # all values should be NaN
            assert inp[over_limit_mask].isnan().all()
