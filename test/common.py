import hypothesis.strategies as st
import os
import pytest
from packaging import version
import torch

# Condition for mark.xfail
CONDITION = version.parse(torch.__version__) == version.parse('1.2') and os.environ.get('PYTORCH_JIT', '1') == '0'
REASON = 'Known bug to Pytorch 1.2.0 with JIT disabled'
check_expected_fail = pytest.mark.xfail(CONDITION, reason=REASON, raises=RuntimeError)

# Set Constants
RTOL = 0
ATOL = 1e-23

FP_BIT_WIDTH = 32

# Define custom type of floating point generator.
# We are never interested in NaN and Infinity. In some case, such as when generating gradients, we may also want to
# exclude zero. For scale factor, we want only positive numbers
float_st = st.floats(allow_nan=False, allow_infinity=False, width=FP_BIT_WIDTH)
float_st_nz = st.floats(allow_nan=False, allow_infinity=False, width=FP_BIT_WIDTH).filter(lambda x: x != 0.0)
float_st_p = st.floats(min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False, width=FP_BIT_WIDTH)
list_float_st = st.lists(float_st, min_size=1)


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
        st.floats(allow_infinity=False, allow_nan=False, width=FP_BIT_WIDTH, min_value=minimum))
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
            st.floats(allow_infinity=False, allow_nan=False, width=FP_BIT_WIDTH, min_value=minimum))
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
            st.floats(allow_infinity=False, allow_nan=False, width=FP_BIT_WIDTH, min_value=minimum))
        minimum_list[i] = minimum
        maximum_list[i] = maximum
    values = draw(st.lists(float_st, min_size=size, max_size=size))
    grad = draw(st.lists(float_st_nz, min_size=size, max_size=size))
    return minimum_list, values, grad, maximum_list
