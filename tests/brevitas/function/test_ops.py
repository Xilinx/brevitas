# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import numpy as np
import pytest
from torch import tensor

from brevitas.function.ops import *
from tests.brevitas.common import assert_allclose
from tests.brevitas.common import BOOLS
from tests.brevitas.common import INT_BIT_WIDTH_TO_TEST
from tests.brevitas.function.hyp_helper import *


@given(x=tensor_clamp_test_st())
def test_tensor_clamp(x):
    """
    Test that all output values of tensor_clamp are bounded by min_val_t and max_val_t and that
    all output values are either the original values or min_val_t or max_val_t.
    """
    min_val, max_val, val = x
    output = tensor_clamp(val, min_val, max_val)
    assert (output <= max_val).all()
    assert (output >= min_val).all()
    assert ((output == min_val) | (output == max_val) | (output == val)).all()


@given(x=tensor_clamp_test_st())
def test_inplace_tensor_clamp(x):
    """
    Test that the output of tensor_clamp_ is equal to the output of tensor_clamp.
    """
    min_val, max_val, val = x
    oop_output = tensor_clamp(val, min_val, max_val)
    tensor_clamp_(val, min_val, max_val)
    assert (oop_output == val).all()


@given(x=binary_sign_test_st())
def test_binary_sign(x):
    """
    Test that all outputs are either 1.0 or -1.0
    """
    output = binary_sign(x)
    assert ((output == tensor(1.0)) | (output == tensor(-1.0))).all()


@given(x=float_tensor_random_shape_st())
def test_round_to_zero(x):
    """
    Test round_to_zero against np.fix on random float tensors
    """
    output = round_to_zero(x)
    reference = torch.from_numpy(np.fix(x.numpy()))
    assert_allclose(output, reference)


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
def test_narrow_range_unsigned_max_int(bit_width):
    """
    Test that max_int unsigned with narrow range enabled is off by 1 wrt narrow range disabled.
    """
    val_true = max_int(signed=False, narrow_range=True, bit_width=tensor(bit_width))
    val_false = max_int(signed=False, narrow_range=False, bit_width=tensor(bit_width))
    assert val_false - val_true == tensor(1)


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_max_int(bit_width):
    """
    Test that max_int signed with narrow range enabled is equal to narrow range disabled.
    """
    val_true = max_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_false = max_int(signed=True, narrow_range=False, bit_width=tensor(bit_width))
    assert val_false == val_true


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_min_int(bit_width):
    """
    Test that min_int signed with narrow range enabled is off by 1 wrt narrow range disabled.
    """
    val_true = min_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_false = min_int(signed=True, narrow_range=False, bit_width=tensor(bit_width))
    assert val_true - val_false == tensor(1)


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
def test_unsigned_min_int(bit_width):
    """
    Test that min_int unsigned is always 0 indipendently of narrow_range.
    """
    val_true = min_int(signed=False, narrow_range=True, bit_width=tensor(bit_width))
    val_false = min_int(signed=False, narrow_range=False, bit_width=tensor(bit_width))
    assert val_true == val_false == tensor(0)


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_symmetric(bit_width):
    """
    Test that narrow_range signed is symmetric around zero.
    """
    val_min = min_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_max = max_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    assert val_min == -val_max
    assert val_max > tensor(0)
    assert val_min < tensor(0)


@pytest.mark.parametrize('bit_width', INT_BIT_WIDTH_TO_TEST)
@pytest.mark.parametrize('signed', BOOLS)
@pytest.mark.parametrize('narrow_range', BOOLS)
def test_narrow_range_signed_interval(bit_width, signed, narrow_range):
    """
    Test the width of interval with and without narrow range enabled.
    """
    val_min = min_int(signed=signed, narrow_range=narrow_range, bit_width=tensor(bit_width))
    val_max = max_int(signed=signed, narrow_range=narrow_range, bit_width=tensor(bit_width))
    if narrow_range:
        assert val_max - val_min == tensor((2 ** bit_width) - 2)
    else:
        assert val_max - val_min == tensor((2 ** bit_width) - 1)
