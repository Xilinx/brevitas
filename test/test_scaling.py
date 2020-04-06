from hypothesis import given, assume, example, note
from hypothesis import strategies as st
from common import *
from brevitas.core.scaling import *
from brevitas.core import ZERO_HW_SENTINEL_VALUE
import pytest
from brevitas.core.stats import StatsOp
from brevitas.function.ops import min_int, max_int


# Constants
OUTCHANNEL = 20
SCALING_MIN_VAL = 0.0
MIN_BIT_WIDTH=1
MAX_BIT_WIDTH=8

# Pytest Parametrize options
restrict_scaling_type_options = [(RestrictValueType.FP), (RestrictValueType.LOG_FP), (RestrictValueType.POWER_OF_TWO)]
is_parameter_bool_options = [(True), (False)]
max_dim_options = [(2), (3), (4)]


# Perform test for standalone scaling. It takes the scaling Tensor (which can be either a scalar or a real Tensor), and
# all the other configuration for StandaloneScaling and RestrictValue.
# Properties tested:
#     - The Value returned by StandaloneScaling has the shape that we expect
#     - Its value is correct, both when scaling.shape == shape, and when scaling is a scalar
# Assumptions:
#     - The implementation of RestrictValue is correct
# Known Issues:
#     - If the value of scaling is too high, then the tests fail cause RestrictValue returns inf
def perform_test_standalonescaling(scaling, shape, restrict_scaling_type, is_parameter):
    expected_restrict_value = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL, SCALING_MIN_VAL)
    expected_restrict_value_op = RestrictValue.restrict_value_op(restrict_scaling_type,
                                                                 restrict_value_op_impl_type=RestrictValueOpImplType.TORCH_FN)
    obj = StandaloneScaling(scaling_init=scaling, is_parameter=is_parameter,
                            parameter_shape=shape, scaling_min_val=SCALING_MIN_VAL,
                            restrict_scaling_type=restrict_scaling_type)
    output = obj(torch.tensor(ZERO_HW_SENTINEL_VALUE))

    expected_output = expected_restrict_value_op(scaling)
    expected_output = expected_restrict_value(expected_output)
    result = True
    if scaling.dim() == 0:
        for element in output.view(-1):
            result = result and (torch.abs(expected_output-element) <= ATOL)
    else:
        result = (torch.allclose(expected_output, output, RTOL, ATOL))
    assert result
    assert output.shape == shape


# Perform test for standalone scaling. It takes the scaling Tensor (which can be either a scalar or a real Tensor), and
# all the other configuration for StandaloneScaling and RestrictValue.
# Properties tested:
#     - The value returned is correct is correct
# Assumptions:
#     - The implementation of RestrictValue is correct
# Known Issues:
#     - If the value of scaling is too high, then the tests fail cause RestrictValue returns inf
def perform_test_statsscaling(scaling, shape, restrict_scaling_type):
    expected_restrict_value = RestrictValue(restrict_scaling_type, FloatToIntImplType.CEIL, SCALING_MIN_VAL)
    expected_restrict_value_op = RestrictValue.restrict_value_op(restrict_scaling_type,
                                                                 restrict_value_op_impl_type=RestrictValueOpImplType.TORCH_FN)
    obj = StatsScaling(stats_op=StatsOp.MAX, restrict_scaling_type=restrict_scaling_type,
                       stats_output_shape=shape, scaling_min_val=SCALING_MIN_VAL, affine=False)
    output = obj(scaling)

    expected_output = expected_restrict_value_op(scaling)
    expected_output = expected_restrict_value(expected_output)
    assert torch.allclose(expected_output, output, RTOL, ATOL)


# Test the case where both scaling_init and the parameter shape are scalar
@given(scaling_init=float_st_p)
@pytest.mark.parametrize('restrict_scaling_type', restrict_scaling_type_options)
@pytest.mark.parametrize('is_parameter', is_parameter_bool_options)
@check_expected_pyt_120_fail
def test_standlonescaling_scalar_scalar(scaling_init, restrict_scaling_type, is_parameter):
    scaling_init = torch.tensor(scaling_init)
    parameter_shape = ()
    perform_test_standalonescaling(scaling_init, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where scaling_init is scalar and parameter_shape is a Tensor.
# In this case, the parameter shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
@given(scaling_init=float_st_p)
@pytest.mark.parametrize('restrict_scaling_type', restrict_scaling_type_options)
@pytest.mark.parametrize('is_parameter', is_parameter_bool_options)
@pytest.mark.parametrize('dim', max_dim_options)
@check_expected_pyt_120_fail
def test_standlonescaling_scalar_tensor(scaling_init, restrict_scaling_type, is_parameter, dim):
    scaling_init = torch.tensor(scaling_init)
    for posix in range(0, dim):
        parameter_shape = [1] * dim
        parameter_shape[posix] = OUTCHANNEL
        parameter_shape = tuple(parameter_shape)

        # Skip case not supported by StandaloneScaling which would throw error
        if len(parameter_shape) > 1 and not is_parameter:
            pytest.xfail(reason="Combination not support by Brevitas")
        perform_test_standalonescaling(scaling_init, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where both scaling_init and the parameter shape are tensor
# In this case, the parameter shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
# Scaling_init has the same dimension as the parameter shape
# To allow this, we pass the 'data' strategy to the function, and then we draw elements interactively inside the
# function.
@given(scaling_init=st.lists(float_st_p, min_size=OUTCHANNEL, max_size=OUTCHANNEL))
@pytest.mark.parametrize('restrict_scaling_type', restrict_scaling_type_options)
@pytest.mark.parametrize('is_parameter', is_parameter_bool_options)
@pytest.mark.parametrize('dim', max_dim_options)
@check_expected_pyt_120_fail
def test_standlonescaling_tensor_tensor(scaling_init, restrict_scaling_type, is_parameter, dim):
    for posix in range(0, dim):
        parameter_shape = [1] * dim
        parameter_shape[posix] = OUTCHANNEL
        parameter_shape = tuple(parameter_shape)

        # Skip case not supported by StandaloneScaling which would throw error
        if len(parameter_shape) > 1 and not is_parameter:
            continue
        scaling_init_tensor = torch.tensor(scaling_init).view(parameter_shape)
        perform_test_standalonescaling(scaling_init_tensor, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where scaling_init is a scalar
@given(scaling_init=float_st_p)
@pytest.mark.parametrize('restrict_scaling_type', restrict_scaling_type_options)
@pytest.mark.dependency(name="statsscaling_scalar")
@check_expected_pyt_120_fail
def test_statsscaling_scalar(scaling_init, restrict_scaling_type):
    scaling_init = torch.tensor(scaling_init)
    perform_test_statsscaling(scaling_init, SCALING_SCALAR_SHAPE, restrict_scaling_type)


# Test the case where scaling_init is a Tensor
# In this case, the stats shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
@pytest.mark.dependency(name="statsscaling_tensor")
@given(scaling_init=st.lists(float_st_p, min_size=OUTCHANNEL, max_size=OUTCHANNEL))
@pytest.mark.parametrize('restrict_scaling_type', restrict_scaling_type_options)
@pytest.mark.parametrize('dim', max_dim_options)
@check_expected_pyt_120_fail
def test_statsscaling_tensor(scaling_init, restrict_scaling_type, dim):
    for posix in range(0, dim):
        shape = [1] * dim
        shape[posix] = OUTCHANNEL
        shape = tuple(shape)
        scaling_init_tensor = torch.tensor(scaling_init).view(shape)
        perform_test_statsscaling(scaling_init_tensor, shape, restrict_scaling_type)


@pytest.mark.dependency(name="runtimestatsscaling", depends=["statsscaling_scalar", "statsscaling_tensor"])
def test_runtimestatsscaling():
    pass


@pytest.mark.dependency(name="parameterstatsscaling", depends=["statsscaling_scalar", "statsscaling_tensor"])
def test_parameterstatsscaling():
    pass


@pytest.mark.dependency(name="signedfpintscale", depends=["result_of_min_int"])
@given(narrow_range=st.booleans(), bit_width=st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH))
@check_expected_pyt_120_fail
def test_signedfpintscale(narrow_range, bit_width):
    obj = SignedFpIntScale(narrow_range)
    bit_width = torch.tensor(bit_width)
    output = obj(bit_width)
    expected_output = -1 * min_int(signed=True, narrow_range=narrow_range, bit_width=bit_width)

    assert output == expected_output


@pytest.mark.dependency(name="unsignedfpintscale", depends=["result_of_max_int"])
def test_unsignedfpintscale():
    pass


@pytest.mark.dependency(name="poweroftwointscale", depends=["result_of_max_int"])
@given(bit_width=st.integers(min_value=MIN_BIT_WIDTH, max_value=MAX_BIT_WIDTH), signed=st.booleans())
@check_expected_pyt_120_fail
def test_poweroftwointscale(bit_width, signed):
    obj = PowerOfTwoIntScale(signed)
    bit_width = torch.tensor(bit_width)
    output = obj(bit_width)
    expected_output = max_int(signed, bit_width)+1

    assert output == expected_output


@pytest.mark.dependency(name="intscaling", depends=["poweroftwointscale", "unsignedfpintscale", "signedfpintscale"])
def test_intscaling():
    pass