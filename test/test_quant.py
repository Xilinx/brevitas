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
from brevitas.core.quant import BinaryQuant, ClampedBinaryQuant, TernaryQuant
from brevitas.core.quant import PrescaledRestrictIntQuantWithInputBitWidth, PrescaledIntQuant, PrescaledRestrictIntQuant
from brevitas.core.quant import RescalingIntQuant
from brevitas.core.quant import IntQuant
from brevitas.function.ops import min_int, max_int
from brevitas.function.ops import tensor_clamp
from brevitas.core.function_wrapper import TensorClamp, RoundSte, FloorSte, CeilSte
import hypothesis.strategies as st
from hypothesis import given
from common import float_st, float_st_nz, two_lists_equal_size, list_float_st, float_st_p
from common import ATOL, RTOL
from unittest.mock import Mock
from brevitas.core import ZERO_HW_SENTINEL_VALUE


# EXECUTE WITH ENVIRONMENTAL VARIABLE PYTORCH_JIT=0

# Used for BinaryQuant and ClampedBinaryQuant. The two tests are basically identical.
def perform_test_binary(binary_type, inp, scaling):
    scaling_impl_mock = Mock()
    scaling_impl_mock.return_value = torch.tensor(scaling)

    obj = binary_type(scaling_impl_mock)
    output, _, _ = obj(inp, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    expected_output = torch.tensor([-1 * scaling, 1 * scaling])

    # Check that output values match one of the possible values in expected_output
    return check_admissible_values(output, expected_output) and check_binary_sign(inp, output)


# Check that the output values are one of the admissible values. If not, return False and fail the test
def check_admissible_values(predicted, admissible):
    result = True
    for value in predicted:
        if value not in admissible:
            result = False

    return result


# Check that after binarization the sign match the one expected given the input value
def check_binary_sign(inp, predicted):
    for value_input, value_predicted in zip(inp, predicted):
        if ((value_input >= 0) and value_predicted < 0) or ((value_input < 0) and value_predicted > 0):
            return False
    return True


# Check that the sign is correct in ternarization. Take in account the margin introduced by the threshold
# Threshold here is overloaded. It corresponds to threshold * scale
def check_ternary_sign(inp, predicted, threshold):
    for value_input, value_predicted in zip(inp, predicted):
        if (value_input.abs() < threshold and value_predicted != 0) or \
           (value_input > threshold and value_predicted < 0) or (value_input < threshold and value_predicted > 0):
            return False

    return True

# Check that all values generated are either -scale or +scale. Check that the sign is coherent
@given(x=list_float_st, scale=float_st_p)
def test_of_BinaryQuant(x, scale):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    assert perform_test_binary(BinaryQuant, value, scale)


# Check that all values generated are either -scale or +scale. Check that the sign is coherent
@given(x=list_float_st, scale=float_st_p)
def test_of_ClampedBinaryQuant(x, scale):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    assert perform_test_binary(ClampedBinaryQuant, value, scale)


# Check that all values generated are either -scale, +scale or 0. Check that the sign is coherent
@given(x=list_float_st, threshold=float_st_p.filter(lambda x: x < 1), scale=float_st_p)
def test_of_TernaryQuant(x, threshold, scale):
    value = torch.tensor(x)

    scale_impl_mock = Mock()
    scale_impl_mock.return_value = torch.tensor(scale)

    obj = TernaryQuant(scale_impl_mock, threshold)
    output, _, _ = obj(value, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    expected_output = torch.tensor([-1 * scale, 1 * scale, 0])

    assert check_admissible_values(output, expected_output) and check_ternary_sign(value, output, scale * threshold)


# Propriety tested:
#  - All input values, once converted to int, are in the admissible range of integers given by the combination of
#    sign, narrow range and bit_width
#  - After apply IntQuant, the result is in a certain range with respected to the clamped floating point version.
#    The range is determined by scale, and varies according to the type of float_to_int function used
# Assumption:
#  - TensorClamp() is used as tensor_clamp_impl. The alternative would be TensorClampSte() whose only difference lies in
#    backpropagation of the gradient, which is something we are not interested in here.
#  - float_to_int implementation is a stripped down version of RestrictValue object for clarity and easiness of testing.
@given(x=list_float_st, narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=2, max_value=8),
       scale=float_st_p, int_scale=st.integers(min_value=1, max_value=256))
def test_IntQuant(x, narrow_range, signed, bit_width, scale, int_scale):
    float_to_int_impl_mock = Mock()
    tensor_clamp_impl = TensorClamp()

    value = torch.tensor(x)
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    scale = torch.tensor(scale)
    int_scale = torch.tensor(int_scale)

    implementation_type = [FloorSte(), CeilSte(), RoundSte()]
    atol_value = [scale, scale, scale/2.0]

    for tol, type in zip(atol_value, implementation_type):
        float_to_int_impl_mock.side_effect = type

        obj = IntQuant(narrow_range=narrow_range, signed=signed, float_to_int_impl=float_to_int_impl_mock,
                       tensor_clamp_impl=tensor_clamp_impl)
        output = obj(scale, int_scale, bit_width, value)

        min_value = int(min_int(signed, narrow_range, bit_width))
        max_value = int(max_int(signed, bit_width))
        admissible_values = [x for x in range(min_value, max_value+1)]

        value = (value / scale) * int_scale
        expected_output = tensor_clamp(value, min_val=min_int(signed, narrow_range, bit_width),
                                       max_val=max_int(signed, bit_width))
        expected_output = (expected_output / int_scale) * scale

        int_output = obj.to_int(scale, int_scale, bit_width, value)
        assert check_admissible_values(int_output, admissible_values)
        assert (torch.allclose(expected_output, output, RTOL, tol))


# Propriety tested:
#  - This function is just a wrapper around IntQuant. For this reason, we are testing only the interface
# Assumption:
#  - IntQuant doesn't perform any real operation and behaves as an identity.
#  - Since IntQuant is created inside the object, we assume that IntQuant is correct, and create an "expected"
#  - (i.e. correct by definition) version for genereting the expected_output. IntQuant is tested in an apposite function
#  - msb_clamp_bit_width returns a random integer between 2 and 8
@given(x=list_float_st, narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=2, max_value=8), scale=float_st_p)
def test_PrescaledRestrictIntQuantWithInputBitWidth(x, narrow_range, signed, scale, bit_width):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    tensor_clamp_impl = TensorClamp()

    msb_clamp_bitwidth_mock = Mock()
    msb_clamp_bitwidth_mock.return_value = torch.tensor(bit_width, dtype=torch.float)
    float_to_int_impl_mock = Mock()
    float_to_int_impl_mock.side_effect = (lambda y: y)

    obj = PrescaledRestrictIntQuantWithInputBitWidth(narrow_range=narrow_range, signed=signed,
                                                     tensor_clamp_impl=tensor_clamp_impl,
                                                     msb_clamp_bit_width_impl=msb_clamp_bitwidth_mock,
                                                     float_to_int_impl=float_to_int_impl_mock)

    output, scale, bit_width = obj(value, scale, bit_width, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    expected_IntQuant = IntQuant(signed=signed, narrow_range=narrow_range, tensor_clamp_impl=tensor_clamp_impl,
                                 float_to_int_impl=float_to_int_impl_mock)
    expected_output = expected_IntQuant(scale, torch.tensor(ZERO_HW_SENTINEL_VALUE) + 1, bit_width, value)

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Propriety tested:
#  - This function is just a wrapper around IntQuant. For this reason, we are testing only the interface
# Assumption:
#  - IntQuant doesn't perform any real operation and behaves as an identity.
#  - Since IntQuant is created inside the object, we assume that IntQuant is correct, and create an "expected"
#    (i.e. correct by definition) version for genereting the expected_output. IntQuant is tested in an apposite function
#  - msb_clamp_bit_width returns a random integer between 2 and 8
@given(x=list_float_st, narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=2, max_value=8), scale=float_st_p)
def test_PrescaledRestrictIntQuanth(x, narrow_range, signed, scale, bit_width):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    bit_width =torch.tensor(bit_width, dtype=torch.float)
    tensor_clamp_impl = TensorClamp()

    msb_clamp_bitwidth_mock = Mock()
    msb_clamp_bitwidth_mock.return_value =bit_width
    float_to_int_impl_mock = Mock()
    float_to_int_impl_mock.side_effect = (lambda y: y)

    obj = PrescaledRestrictIntQuant(narrow_range=narrow_range, signed=signed,
                                    tensor_clamp_impl=tensor_clamp_impl,
                                    msb_clamp_bit_width_impl=msb_clamp_bitwidth_mock,
                                    float_to_int_impl=float_to_int_impl_mock)

    output, scale, bit_width = obj(value, scale, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    expected_IntQuant = IntQuant(signed=signed, narrow_range=narrow_range, tensor_clamp_impl=tensor_clamp_impl,
                                 float_to_int_impl=float_to_int_impl_mock)
    expected_output = expected_IntQuant(scale, torch.tensor(ZERO_HW_SENTINEL_VALUE) + 1, bit_width, value)

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Propriety tested:
#  - This function is just a wrapper around IntQuant. For this reason, we are testing only the interface
# Assumption:
#  - IntQuant doesn't perform any real operation and behaves as an identity.
#  - Since IntQuant is created inside the object, we assume that IntQuant is correct, and create an "expected"
#  - (i.e. correct by definition) version for genereting the expected_output. IntQuant is tested in an apposite function
@given(x=list_float_st, narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=2, max_value=8), scale=float_st_p)
def test_PrescaledIntQuant(x, narrow_range, signed, scale, bit_width):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    tensor_clamp_impl = TensorClamp()

    float_to_int_impl_mock = Mock()
    float_to_int_impl_mock.side_effect = (lambda y: y)

    obj = PrescaledIntQuant(narrow_range=narrow_range, signed=signed,
                            tensor_clamp_impl=tensor_clamp_impl,
                            float_to_int_impl=float_to_int_impl_mock)

    output, scale, bit_width = obj(value, scale, bit_width, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    expected_IntQuant = IntQuant(signed=signed, narrow_range=narrow_range, tensor_clamp_impl=tensor_clamp_impl,
                                 float_to_int_impl=float_to_int_impl_mock)
    expected_output = expected_IntQuant(scale, torch.tensor(ZERO_HW_SENTINEL_VALUE) + 1, bit_width, value)

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


# Propriety tested:
#  - This function is just a wrapper around IntQuant. For this reason, we are testing the interface
#  - In addition, we test that the scale factor generated is correct with what we expect.
# Assumption:
#  - IntQuant doesn't perform any real operation and behaves as an identity.
#  - Since IntQuant is created inside the object, we assume that IntQuant is correct, and create an "expected"
#  - (i.e. correct by definition) version for genereting the expected_output. IntQuant is tested in an apposite function
#  - Runtime is default to true, since it has no effect in the generation of the scale factor (which is random)
#  - msb_clamp_bit_width returns a random integer between 2 and 8
@given(x=list_float_st, narrow_range=st.booleans(), signed=st.booleans(),
       bit_width=st.integers(min_value=2, max_value=8), scale=float_st_p,
       int_scale=st.integers(min_value=1, max_value=256))
def test_RescalingIntQuant(x, narrow_range, signed, scale, int_scale, bit_width):
    value = torch.tensor(x)
    scale = torch.tensor(scale)
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    int_scale = torch.tensor(int_scale, dtype=torch.float)
    tensor_clamp_impl = TensorClamp()

    msb_clamp_bitwidth_mock = Mock()
    msb_clamp_bitwidth_mock.return_value = bit_width
    float_to_int_impl_mock = Mock()
    float_to_int_impl_mock.side_effect = (lambda y: y)
    int_scaling_impl_mock = Mock()
    int_scaling_impl_mock.return_value = int_scale
    scaling_impl = Mock()
    scaling_impl.return_value = scale

    obj = RescalingIntQuant(narrow_range=narrow_range, signed=signed,
                            runtime=True,
                            tensor_clamp_impl=tensor_clamp_impl,
                            float_to_int_impl=float_to_int_impl_mock,
                            int_scaling_impl=int_scaling_impl_mock,
                            msb_clamp_bit_width_impl=msb_clamp_bitwidth_mock,
                            scaling_impl=scaling_impl)

    output, scale_out, bit_width = obj(value, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    expected_IntQuant = IntQuant(signed=signed, narrow_range=narrow_range, tensor_clamp_impl=tensor_clamp_impl,
                                 float_to_int_impl=float_to_int_impl_mock)
    expected_output = expected_IntQuant(scale, int_scale, bit_width, value)
    expected_scale = scale/int_scale
    assert (torch.allclose(expected_output, output, RTOL, ATOL))
    assert (torch.allclose(expected_scale, scale_out, RTOL, ATOL))
