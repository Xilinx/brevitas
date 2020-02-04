import hypothesis.strategies as st


# Set Constants
RTOL = 1e-03
ATOL = 1e-03


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
