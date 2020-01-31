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

import torch
import math
from brevitas.function.ops import *
from brevitas.function.ops_ste import *
import pytest
import hypothesis.strategies as st
from hypothesis import assume, given, note, example
from hypothesis import seed as set_seed
from hypothesis import settings, HealthCheck

# Remove Hypothesis check for too slow tests. Some tests requires particular input conditions, and it may take a
# while to generate the appropriate inputs.
settings.register_profile("standard", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("standard")

# Set Constants
SEED = 123456
RTOL = 1e-03
ATOL = 1e-03
torch.random.manual_seed(SEED)
set_seed(SEED)


# Define custom type of floating point generator.
# We are never interested in NaN and Infinity. In some case, such as when generating gradients, we may also want to
# exclude zero
float_st = st.floats(allow_nan=False, allow_infinity=False, width=32)
float_st_nz = st.floats(allow_nan=False, allow_infinity=False, width=32).filter(lambda x: x != 0.0)


# Create custom strategy for generating two lists of floats with equal size
# This is used every time we need to test the STE property of an operation. We need an input as list of floats, and
# a gradient as list of floats with zero filtered out
@st.composite
def two_lists_equal_size(draw):
    list_one = draw(st.lists(float_st, min_size=1))
    size = len(list_one)
    list_two = draw(st.lists(float_st_nz, min_size=size, max_size=size))
    return list_one, list_two


# Create custom strategy for generating three floating point numbers such that minimum < value < maximum
# Used for Clamps function
@st.composite
def two_ordered_numbers(draw):
    minimum = draw(float_st)
    maximum = draw(
        st.floats(allow_infinity=False, allow_nan=False, width=32, min_value=minimum))
    return minimum, maximum


# Ad-Hoc strategy for tensor clamp input
# We need to generate three lists of the same size. Max and Min lists are created such that
# Min[i] <= Max[i] for i =1...10, where 10 is the length of the lists (fixed for timing reasons)
@st.composite
def tensor_clamp_input(draw):
    size = 10
    minimum_list = [0] * size
    maximum_list = [0] * size
    for i in range(size):
        minimum = draw(float_st)
        maximum = draw(
            st.floats(allow_infinity=False, allow_nan=False, width=32, min_value=minimum))
        minimum_list[i] = minimum
        maximum_list[i] = maximum
    values = draw(st.lists(float_st, min_size=size, max_size=size))
    return minimum_list, values, maximum_list


# Same as tensor_clamp_input. In this case there is a fourth list, the Gradient List, which has the same size of the
# other threes and that doesn't include zeroes
@st.composite
def tensor_clamp_ste_input(draw):
    size = 10
    minimum_list = [0] * size
    maximum_list = [0] * size
    for i in range(size):
        minimum = draw(float_st)
        maximum = draw(
            st.floats(allow_infinity=False, allow_nan=False, width=32, min_value=minimum))
        minimum_list[i] = minimum
        maximum_list[i] = maximum
    values = draw(st.lists(float_st, min_size=size, max_size=size))
    grad = draw(st.lists(float_st_nz, min_size=size, max_size=size))
    return minimum_list, values, grad, maximum_list


# When testing STE attribute, we pass an external gradient to backward and we check that it is correctly backpropagated
# over the function
@given(x=two_lists_equal_size())
@settings(deadline=None) # Removes timing check
def test_ste_of_round_ste(x):
    value = x[0]
    grad = x[1]
    value = torch.tensor(value, requires_grad=True)
    grad = torch.tensor(grad)

    output = round_ste(value)
    output.backward(grad, retain_graph=True)

    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


# Generate a list of custom floats with at least one element
@given(x=st.lists(float_st, min_size=1))
def test_result_of_round_ste(x):
    x = torch.tensor(x)

    output = round_ste(x)
    expected_output = torch.round(x)

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=tensor_clamp_input())
def test_result_of_tensor_clamp(x):
    minimum = torch.tensor(x[0])
    value = torch.tensor(x[1])
    maximum = torch.tensor(x[2])

    output = tensor_clamp(value, minimum, maximum)
    expected_output = []
    for i in range(minimum.size()[0]):
        expected_output.append(torch.clamp(value[i], x[0][i], x[2][i]))
    expected_output = torch.tensor(expected_output)

    assert ((output >= minimum).all() and (output <= maximum).all())
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=tensor_clamp_ste_input())
def test_ste_of_tensor_clamp_ste(x):
    minimum = torch.tensor(x[0])
    value = torch.tensor(x[1], requires_grad=True)
    grad = x[2]
    grad = torch.tensor(grad)
    maximum = torch.tensor(x[3])

    output = tensor_clamp_ste(value, minimum, maximum)

    output.backward(grad, retain_graph=True)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))

# Test different combinations of Narrow Range (True/False) and BitWidth (1...8)
@given(narrow_range=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_max_uint(narrow_range, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = max_uint(narrow_range, bit_width)

    if narrow_range:
        expected_output = (2 ** bit_width) - 2
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Test different combinations of Narrow Range (True/False) and BitWidth (1...8)
@given(signed=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_max_int(signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = max_int(signed, bit_width)

    if signed:
        expected_output = (2 ** (bit_width - 1)) - 1
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Test different combinations of Narrow Range (True/False), Signed (True/False), and BitWidth (1...8)
@given(narrow_range=st.booleans(), signed=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_min_int(narrow_range, signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = min_int(signed, narrow_range, bit_width)

    if signed and narrow_range:
        expected_output = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        expected_output = -(2 ** (bit_width - 1))
    else:
        expected_output = torch.tensor(0.0)

    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Requires two floats as maximum and minimum and a tensor of float
@given(minmax=two_ordered_numbers(), x=st.lists(float_st, min_size=1))
def test_result_of_scalar_clamp_ste(minmax, x):
    minimum = torch.tensor(minmax[0])
    value = torch.tensor(x)
    maximum = torch.tensor(minmax[1])

    output = scalar_clamp_ste(value, minimum, maximum)
    expected_output = torch.clamp(value, minmax[0], minmax[1])

    assert ((output >= minimum).all() and (output <= maximum).all())
    assert (torch.allclose(expected_output, output, RTOL, ATOL))

# Same as before, but with two Tensors of Float
@given(minmax=two_ordered_numbers(), x=two_lists_equal_size())
def test_ste_of_scalar_clamp_ste(minmax, x):
    minimum = minmax[0]
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])
    maximum = minmax[1]

    output = scalar_clamp_ste(value, minimum, maximum)

    output.backward(grad, retain_graph=True)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


@given(x=st.lists(float_st, min_size=1))
def test_result_of_ceil_ste(x):
    value = torch.tensor(x)
    output = ceil_ste(value)
    expected_output = torch.ceil(value)
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=two_lists_equal_size())
def test_ste_of_ceil_ste(x):
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])

    output = ceil_ste(value)
    output.backward(grad)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


@given(x=st.lists(float_st, min_size=1))
def test_result_of_floor_ste(x):
    value = torch.tensor(x)
    output = floor_ste(value)
    expected_output = torch.floor(value)
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=two_lists_equal_size())
def test_ste_of_floor_ste(x):
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])

    output = floor_ste(value)
    output.backward(grad)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


@given(x=st.lists(float_st, min_size=1))
def test_result_of_binary_sign_ste(x):
    value = torch.tensor(x)
    output = binary_sign_ste(value)
    positive_mask = torch.ge(value, 0.0)
    negative_mask = torch.lt(value, 0.0)
    expected_output = positive_mask.float() - negative_mask.float()
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=two_lists_equal_size())
def test_ste_of_binary_sign_ste(x):
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])
    output = binary_sign_ste(value)
    output.backward(grad)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


@given(x=st.lists(float_st, min_size=1))
def test_result_of_ternary_sign_ste(x):
    value = torch.tensor(x)
    output = ternary_sign_ste(value)
    expected_output = torch.sign(value)
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(x=two_lists_equal_size())
def test_ste_of_ternary_sign_ste(x):
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])
    output = ternary_sign_ste(value)
    output.backward(grad)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))
