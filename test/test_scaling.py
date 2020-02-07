from hypothesis import given, assume, example, note
from hypothesis import strategies as st
from common import *
from brevitas.core.scaling import *
from brevitas.core import ZERO_HW_SENTINEL_VALUE
import itertools


from brevitas.core.stats import StatsOp

OUTCHANNEL = 20
SCALING_MIN_VAL = 0.0


# Generate configurations for StandaloneScaling. It takes as input isscalar bool, which indicates whether the parameter
# shape is a scalar or not. Isscalar doesn't refer to the dimension of scaling_init.
def generator_standalonescaling(isscalar):
    maxDim = [2, 3, 4]
    restrict_scaling_type_comb = [RestrictValueType.FP, RestrictValueType.LOG_FP, RestrictValueType.POWER_OF_TWO]
    is_parameter_bool = [True, False]
    if isscalar:
        combinations = [restrict_scaling_type_comb, is_parameter_bool]
        for restrict_scaling_type, is_parameter in list(itertools.product(*combinations)):
            yield (restrict_scaling_type, is_parameter)
    else:
        combinations = [maxDim, restrict_scaling_type_comb, is_parameter_bool]
        for dim, restrict_scaling_type, is_parameter in list(itertools.product(*combinations)):
            yield (dim, restrict_scaling_type, is_parameter)


# Generate configurations for StatsScaling. It takes as input isscalar bool, which indicates whether the stats shape is
# a scalar or not
def generator_statsscaling(isscalar):
    maxDim = [2, 3, 4]
    restrict_scaling_type_comb = [RestrictValueType.FP, RestrictValueType.LOG_FP, RestrictValueType.POWER_OF_TWO]
    if isscalar:
        for restrict_scaling_type in restrict_scaling_type_comb:
            yield restrict_scaling_type
    else:
        combinations = [maxDim, restrict_scaling_type_comb]
        for dim, restrict_scaling_type in list(itertools.product(*combinations)):
            for posix in range(0, dim):
                stats_shape = [1] * dim
                stats_shape[posix] = OUTCHANNEL
                stats_shape = tuple(stats_shape)
                yield (stats_shape, restrict_scaling_type)


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
    result = result and (output.shape == shape)
    return result


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
    return torch.allclose(expected_output, output, RTOL, ATOL)


# Test the case where both scaling_init and the parameter shape are scalar
@given(scaling_init=float_st_p)
def test_standlonescaling_scalar_scalar(scaling_init):
    scaling_init = torch.tensor(scaling_init)
    parameter_shape = ()
    isscalar = True
    for restrict_scaling_type, is_parameter in generator_standalonescaling(isscalar):
        assert perform_test_standalonescaling(scaling_init, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where scaling_init is scalar and parameter_shape is a Tensor.
# In this case, the parameter shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
@given(scaling_init=float_st_p)
def test_standlonescaling_scalar_tensor(scaling_init):
    scaling_init = torch.tensor(scaling_init)
    isscalar = False
    for dim, restrict_scaling_type, is_parameter in generator_standalonescaling(isscalar):
        for posix in range(0, dim):
            parameter_shape = [1] * dim
            parameter_shape[posix] = OUTCHANNEL
            parameter_shape = tuple(parameter_shape)

            # Skip case not supported by StandaloneScaling which would throw error
            if len(parameter_shape) > 1 and not is_parameter:
                continue
            assert perform_test_standalonescaling(scaling_init, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where both scaling_init and the parameter shape are tensor
# In this case, the parameter shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
# Scaling_init has the same dimension as the parameter shape
# To allow this, we pass the 'data' strategy to the function, and then we draw elements interactively inside the
# function.
@given(scaling_init=st.lists(float_st_p, min_size=OUTCHANNEL, max_size=OUTCHANNEL))
def test_standlonescaling_tensor_tensor(scaling_init):
    isscalar = False
    for dim, restrict_scaling_type, is_parameter in generator_standalonescaling(isscalar):
        for posix in range(0, dim):
            parameter_shape = [1] * dim
            parameter_shape[posix] = OUTCHANNEL
            parameter_shape = tuple(parameter_shape)
            scaling_init_tensor = torch.ones(OUTCHANNEL)

            # Skip case not supported by StandaloneScaling which would throw error
            if len(parameter_shape) > 1 and not is_parameter:
                continue
            scaling_init_tensor = torch.tensor(scaling_init).view(parameter_shape)
            assert perform_test_standalonescaling(scaling_init_tensor, parameter_shape, restrict_scaling_type, is_parameter)


# Test the case where scaling_init is a scalar
@given(scaling_init=float_st_p)
def test_statsscaling_scalar(scaling_init):
    scaling_init = torch.tensor(scaling_init)
    for restrict_scaling_type in generator_statsscaling(True):
        assert perform_test_statsscaling(scaling_init, SCALING_SCALAR_SHAPE, restrict_scaling_type)


# Test the case where scaling_init is a Tensor
# In this case, the stats shape has 2,3, or 4 dimensions. All the dimensions are set to 1, with the exception of one
# dimension. All possible combinations are tested.
@given(scaling_init=st.lists(float_st_p, min_size=OUTCHANNEL, max_size=OUTCHANNEL))
def test_statsscaling_tensor(scaling_init):
    for shape, restrict_scaling_type in generator_statsscaling(False):
        scaling_init_tensor = torch.tensor(scaling_init).view(shape)
        assert perform_test_statsscaling(scaling_init_tensor, shape, restrict_scaling_type)

