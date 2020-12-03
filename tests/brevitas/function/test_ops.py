# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from hypothesis import given
import pytest

import numpy as np
from torch import tensor

from brevitas.function.ops import *

from tests.brevitas.common import MIN_INT_BIT_WIDTH, MAX_INT_BIT_WIDTH, BOOLS, assert_allclose
from tests.brevitas.function.hyp_helper import *


BIT_WIDTH_TO_TEST = range(MIN_INT_BIT_WIDTH, MAX_INT_BIT_WIDTH + 1)


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


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
def test_narrow_range_unsigned_max_int(bit_width):
    """
    Test that max_int unsigned with narrow range enabled is off by 1 wrt narrow range disabled.
    """
    val_true = max_int(signed=False, narrow_range=True, bit_width=tensor(bit_width))
    val_false = max_int(signed=False, narrow_range=False, bit_width=tensor(bit_width))
    assert val_false - val_true == tensor(1)


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_max_int(bit_width):
    """
    Test that max_int signed with narrow range enabled is equal to narrow range disabled.
    """
    val_true = max_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_false = max_int(signed=True, narrow_range=False, bit_width=tensor(bit_width))
    assert val_false == val_true


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_min_int(bit_width):
    """
    Test that min_int signed with narrow range enabled is off by 1 wrt narrow range disabled.
    """
    val_true = min_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_false = min_int(signed=True, narrow_range=False, bit_width=tensor(bit_width))
    assert val_true - val_false == tensor(1)


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
def test_unsigned_min_int(bit_width):
    """
    Test that min_int unsigned is always 0 indipendently of narrow_range.
    """
    val_true = min_int(signed=False, narrow_range=True, bit_width=tensor(bit_width))
    val_false = min_int(signed=False, narrow_range=False, bit_width=tensor(bit_width))
    assert val_true == val_false == tensor(0)


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
def test_narrow_range_signed_symmetric(bit_width):
    """
    Test that narrow_range signed is symmetric around zero.
    """
    val_min = min_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    val_max = max_int(signed=True, narrow_range=True, bit_width=tensor(bit_width))
    assert val_min == - val_max
    assert val_max > tensor(0)
    assert val_min < tensor(0)


@pytest.mark.parametrize('bit_width', BIT_WIDTH_TO_TEST)
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
