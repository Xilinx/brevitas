from operator import mul
from functools import reduce, partial

from hypothesis.strategies import SearchStrategy
from hypothesis import settings, HealthCheck
from hypothesis import seed as set_seed
import hypothesis.strategies as st
import torch

from tests.brevitas.common import FP_BIT_WIDTH
from tests.conftest import SEED

# Remove Hypothesis check for slow tests and function scoped fixtures.
# Some tests requires particular input conditions, and it may take a while to generate them.
# Issues with function scoped fixtures are handled manually on a case-by-case basis.
supress_health_checks = [HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
settings.register_profile("standard", deadline=None, suppress_health_check=supress_health_checks)
settings.load_profile("standard")
set_seed(SEED)


def float_st() -> SearchStrategy:
    """
    Generate a 32 bit float, excluding NaN and infinity
    """
    return st.floats(allow_nan=False, allow_infinity=False, width=FP_BIT_WIDTH)


def float_nz_st(min_val=None, max_val=None) -> SearchStrategy:
    """
    Generate a non zero 32 bit float, excluding NaN and infinity
    """
    floats = st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=FP_BIT_WIDTH,
        min_value=min_val,
        max_value=max_val)
    nz_floats = floats.filter(lambda x: x != 0.0)
    return nz_floats


def float_p_st() -> SearchStrategy:
    """
    Generate a 32 bit positive float, excluding NaN and infinity
    """
    p_floats = st.floats(
        min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False, width=FP_BIT_WIDTH)
    return p_floats


@st.composite
def min_max_list_st(draw, size):
    """
    Generate two list of numbers min_val_list and max_val_list where min_val_list <= max_val_list
    holds element-wise.
    """
    floats = partial(st.floats, allow_infinity=False, allow_nan=False, width=FP_BIT_WIDTH)
    min_val_list = [draw(float_st()) for i in range(size)]
    max_val_list = [draw(floats(min_value=i)) for i in min_val_list]
    return min_val_list, max_val_list


@st.composite
def random_tensor_shape_st(draw, min_dims: int = 1, max_dims: int = 4, max_size: int = 3):
    """
    Generate a random tensor shape (both number of dims and size of each dim).
    """
    dim_size = st.integers(min_value=1, max_value=max_size)
    dims = draw(st.lists(dim_size, min_size=min_dims, max_size=max_dims))
    return dims


@st.composite
def empty_tensor_random_shape_st(draw, min_dims: int = 1, max_dims: int = 4, max_size: int = 3):
    """
    Generate an torch.empty tensor of random shape.
    """
    shape = draw(random_tensor_shape_st(min_dims, max_dims, max_size))
    return torch.empty(*shape)


@st.composite
def float_tensor_st(draw, shape):
    """
    Generate a float tensor of hypothesis-picked values of a given shape.
    """
    size = reduce(mul, shape, 1)
    float_list = draw(st.lists(float_st(), min_size=size, max_size=size))
    t = torch.tensor(float_list).view(shape)
    return t

@st.composite
def float_tensor_nz_st(draw, shape):
    """
    Generate a non-zero float tensor of hypothesis-picked values of a given shape.
    """
    size = reduce(mul, shape, 1)
    float_list = draw(st.lists(float_nz_st(), min_size=size, max_size=size))
    t = torch.tensor(float_list).view(shape)
    return t


@st.composite
def scalar_float_tensor_st(draw):
    """
    Generate a scalar float tensor.
    """
    t = torch.tensor(draw(float_st()))
    return t


@st.composite
def scalar_float_p_tensor_st(draw):
    """
    Generate a positive scalar float tensor.
    """
    t = torch.tensor(draw(float_p_st()))
    return t


@st.composite
def scalar_float_nz_tensor_st(draw, min_val=None, max_val=None):
    """
    Generate a scalar non-zero float tensor.
    """
    t = torch.tensor(float(draw(float_nz_st(min_val=min_val, max_val=max_val))))
    return t


@st.composite
def float_tensor_random_shape_st(draw, min_dims: int = 1, max_dims: int = 4, max_size: int = 3):
    """
    Generate a float tensor of random shape.
    """
    shape = draw(random_tensor_shape_st(min_dims, max_dims, max_size))
    t = draw(float_tensor_st(shape))
    return t


@st.composite
def two_float_tensor_random_shape_st(draw, min_dims=1, max_dims=4, max_size=3):
    """
    Generate a tuple of float tensors of the same random shape.
    """
    shape = draw(random_tensor_shape_st(min_dims, max_dims, max_size))
    size = reduce(mul, shape, 1)
    float_list1 = draw(st.lists(float_st(), min_size=size, max_size=size))
    float_list2 = draw(st.lists(float_st(), min_size=size, max_size=size))
    tensor1 = torch.tensor(float_list1).view(shape)
    tensor2 = torch.tensor(float_list2).view(shape)
    return tensor1, tensor2


@st.composite
def min_max_scalar_tensor_st(draw):
    """
    Generate two scalar tensors such that min_val <= max_val.
    """
    min_val = draw(float_st())
    max_val = draw(
        st.floats(allow_infinity=False, allow_nan=False, width=FP_BIT_WIDTH, min_value=min_val))
    return torch.tensor(min_val), torch.tensor(max_val)


@st.composite
def min_max_tensor_st(draw, shape):
    """
    Generate two tensors of the same shape such that min_tensor <= max_tensor elementwise.
    """
    size = reduce(mul, shape, 1)
    min_list, max_list = draw(min_max_list_st(size))
    min_tensor = torch.tensor(min_list).view(*shape)
    max_tensor = torch.tensor(max_list).view(*shape)
    return min_tensor, max_tensor


@st.composite
def min_max_tensor_random_shape_st(draw, min_dims=1, max_dims=4, max_size=3):
    """
    Generate two tensors of the same random shape such that min_tensor <= max_tensor elementwise.
    """
    shape = draw(random_tensor_shape_st(min_dims, max_dims, max_size))
    size = reduce(mul, shape, 1)
    min_list, max_list = draw(min_max_list_st(size))
    min_tensor = torch.tensor(min_list).view(*shape)
    max_tensor = torch.tensor(max_list).view(*shape)
    return min_tensor, max_tensor