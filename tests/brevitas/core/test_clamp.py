# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import torch

from brevitas.function.ops import max_float
from brevitas.quant.experimental.float import Fp8e4m3Weight
from brevitas.quant.experimental.float import Fp8e5m2Weight
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZWeight
from brevitas.quant.experimental.float_quant_fnuz import Fp8e5m2FNUZWeight
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeight
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPWeight
from brevitas.utils.float_quant_utils import get_max_available_float
from brevitas.utils.float_quant_utils import get_min_available_float
from tests.brevitas.hyp_helper import float_tensor_random_shape_st

from .minifloat_fixtures import *

FORMAT_MAXVAL_MAP = {
    Fp8e5m2OCPWeight: 57344.,
    Fp8e4m3OCPWeight: 448.,
    Fp8e4m3Weight: 480.,
    Fp8e5m2Weight: 114688.,
    Fp8e4m3FNUZWeight: 240.,
    Fp8e5m2FNUZWeight: 57344.,
    Fp8e7m0Weight: 2.0 ** 64,  # Custom exponent_bit_width
    Fp8e6m1Weight: 6442450944.0,
    Fp8e3m4Weight: 31.0,
    Fp8e2m5Weight: 7.875,
    Fp8e1m6Weight: 3.96875}

FORMAT_MINVAL_MAP = {
    Fp8e5m2OCPWeight: 2.0 ** -16,
    Fp8e4m3OCPWeight: 2.0 ** -9,
    Fp8e4m3Weight: 2.0 ** -9,
    Fp8e5m2Weight: 2.0 ** -16,
    Fp8e4m3FNUZWeight: 2.0 ** -10,
    Fp8e5m2FNUZWeight: 2.0 ** -17,
    Fp8e7m0Weight: 2.0 ** -63,  # Custom exponent_bit_width
    Fp8e6m1Weight: 2.0 ** -31,
    Fp8e3m4Weight: 2.0 ** -6,
    Fp8e2m5Weight: 2.0 ** -5,
    Fp8e1m6Weight: 2.0 ** -5}


@pytest.mark.parametrize(
    'minifloat, expected_max_val',
    ((format, max_val) for format, max_val in FORMAT_MAXVAL_MAP.items()))
def test_max_value(minifloat, expected_max_val):
    max_val = max_float(
        torch.tensor(minifloat.exponent_bit_width, dtype=torch.float32),
        torch.tensor(minifloat.mantissa_bit_width, dtype=torch.float32),
        torch.tensor(minifloat.exponent_bias, dtype=torch.float32))
    max_available_float = get_max_available_float(
        minifloat.exponent_bit_width,
        minifloat.mantissa_bit_width,
        minifloat.exponent_bias,
        minifloat.float_clamp_impl.nan_values,
        minifloat.float_clamp_impl.inf_values,
        minifloat.float_clamp_impl.saturating)
    max_available_float = torch.tensor(max_available_float)
    max_val = torch.min(max_val, max_available_float)

    assert expected_max_val == max_val


@pytest.mark.parametrize(
    'minifloat, expected_min_val',
    ((format, min_val) for format, min_val in FORMAT_MINVAL_MAP.items()))
def test_min_value(minifloat, expected_min_val):
    min_val = get_min_available_float(
        minifloat.exponent_bit_width,
        minifloat.mantissa_bit_width,
        minifloat.exponent_bias,
    )

    assert expected_min_val == min_val


@given(inp=float_tensor_random_shape_st())
def test_float_clamp(inp, fp8_clamp):

    max_val = max_float(
        torch.tensor(fp8_clamp.exponent_bit_width, dtype=torch.float32),
        torch.tensor(fp8_clamp.mantissa_bit_width, dtype=torch.float32),
        torch.tensor(fp8_clamp.exponent_bias, dtype=torch.float32))
    max_available_float = get_max_available_float(
        fp8_clamp.exponent_bit_width,
        fp8_clamp.mantissa_bit_width,
        fp8_clamp.exponent_bias,
        fp8_clamp.float_clamp_impl.nan_values,
        fp8_clamp.float_clamp_impl.inf_values,
        fp8_clamp.float_clamp_impl.saturating)
    max_available_float = torch.tensor(max_available_float)
    max_val = torch.min(max_val, max_available_float)
    # get values that exceed max_val
    over_limit_mask = inp.abs() > max_val

    # clamp inp
    inp, *_ = fp8_clamp.float_clamp_impl(
        inp,
        torch.tensor(fp8_clamp.exponent_bit_width),
        torch.tensor(fp8_clamp.mantissa_bit_width),
        torch.tensor(fp8_clamp.exponent_bias))

    if fp8_clamp.float_clamp_impl.saturating:
        # should be clamped to +- max val
        assert (inp[over_limit_mask].abs() == max_val).all()
    else:
        # if inf_values, over limit mask should now be all inf
        if fp8_clamp.float_clamp_impl.inf_values:
            # all values exceeding max_val should be inf
            assert inp[over_limit_mask].isinf().all()
        else:
            # all values should be NaN
            assert inp[over_limit_mask].isnan().all()
