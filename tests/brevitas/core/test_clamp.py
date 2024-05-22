# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import torch

from brevitas.function.ops import max_float
from brevitas.quant.experimental.float import Fp8e4m3Weight
from brevitas.quant.experimental.float import Fp8e5m2Weight
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeight
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPWeight
from brevitas.utils.float_quant_utils import get_max_available_float
from tests.brevitas.hyp_helper import float_tensor_random_shape_st

from .minifloat_fixtures import *

FORMAT_MAXVAL_MAP = {
    Fp8e5m2OCPWeight: 57344., Fp8e4m3OCPWeight: 448., Fp8e4m3Weight: 480., Fp8e5m2Weight: 114688.}


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
