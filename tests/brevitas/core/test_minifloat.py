# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest

from brevitas.quant.experimental.float import Fp8e4m3Weight
from brevitas.quant.experimental.float import Fp8e5m2Weight
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeight
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPWeight
from tests.brevitas.hyp_helper import float_tensor_random_shape_st

from .minifloat_fixtures import *

FORMATS = {
    Fp8e5m2OCPWeight: 57344., Fp8e4m3OCPWeight: 448., Fp8e4m3Weight: 480., Fp8e5m2Weight: 114688.}


@pytest.mark.parametrize(
    'minifloat, expected_max_val', ((format, max_val) for format, max_val in FORMATS.items()))
def test_max_value(minifloat, expected_max_val):
    max_val = minifloat.float_clamp_impl.max_value()

    assert expected_max_val == max_val


@given(inp=float_tensor_random_shape_st())
def test_clamp(inp, fp8_clamp):
    max_val = fp8_clamp.float_clamp_impl.max_value()
    # get values that exceed max_val
    over_limit_mask = inp.abs() > max_val

    # clamp inp
    inp = fp8_clamp.float_clamp_impl(inp)

    if fp8_clamp.float_clamp_impl.saturating:
        # should be clamped to +- max val
        assert (inp[over_limit_mask].abs() == max_val).all()
    else:
        # if inf_values, over limit mask should now be all inf
        if fp8_clamp.float_clamp_impl.has_inf_values:
            # all values exceeding max_val should be inf
            assert inp[over_limit_mask].isinf().all()
        else:
            # all values should be NaN
            assert inp[over_limit_mask].isnan().all()
