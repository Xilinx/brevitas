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

from brevitas.function.ops import *
from brevitas.function.ops_ste import *
from hypothesis import given
from common import *

MIN_BIT_WIDTH = 1
MAX_BIT_WIDTH = 8

# When testing STE attribute, we pass an external gradient to backward and we check that it is correctly backpropagated
# over the function
@given(x=two_lists_equal_size())
def test_ste_of_round_ste(x):
    value = x[0]
    grad = x[1]
    value = torch.tensor(value, requires_grad=True)
    grad = torch.tensor(grad)

    output = round_ste(value)
    output.backward(grad, retain_graph=True)

    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


# Generate a list of custom floats with at least one element
@given(x=list_float_st)
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
@given(narrow_range=st.booleans(), bit_width=st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH))
def test_result_of_max_uint(narrow_range, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = max_uint(narrow_range, bit_width)

    if narrow_range:
        expected_output = (2 ** bit_width) - 2
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = expected_output

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Test different combinations of Narrow Range (True/False) and BitWidth (1...8)
@given(signed=st.booleans(), bit_width=st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH))
def test_result_of_max_int(signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = max_int(signed, bit_width)

    if signed:
        expected_output = (2 ** (bit_width - 1)) - 1
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = expected_output

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Test different combinations of Narrow Range (True/False), Signed (True/False), and BitWidth (1...8)
@given(narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH))
def test_result_of_min_int(narrow_range, signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = min_int(signed, narrow_range, bit_width)

    if signed and narrow_range:
        expected_output = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        expected_output = -(2 ** (bit_width - 1))
    else:
        expected_output = torch.tensor(0.0)

    expected_output = expected_output

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Requires two floats as maximum and minimum and a tensor of float
@given(minmax=two_ordered_numbers(), x=list_float_st)
def test_result_of_scalar_clamp_ste(minmax, x):
    minimum = torch.tensor(minmax[0])
    value = torch.tensor(x)
    maximum = torch.tensor(minmax[1])

    output = scalar_clamp_ste(value, minimum, maximum)
    expected_output = torch.clamp(value, minmax[0], minmax[1])

    assert ((output >= minimum).all() and (output <= maximum).all())
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Same as test_result_of_scalar_clamp_ste, but with two Tensors of Float: one for the input and the other one for the
# gradient
@given(minmax=two_ordered_numbers(), x=two_lists_equal_size())
def test_ste_of_scalar_clamp_ste(minmax, x):
    minimum = minmax[0]
    value = torch.tensor(x[0], requires_grad=True)
    grad = torch.tensor(x[1])
    maximum = minmax[1]

    output = scalar_clamp_ste(value, minimum, maximum)

    output.backward(grad, retain_graph=True)
    assert (torch.allclose(grad, value.grad, RTOL, ATOL))


@given(x=list_float_st)
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


@given(x=list_float_st)
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


@given(x=list_float_st)
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


@given(x=list_float_st)
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
