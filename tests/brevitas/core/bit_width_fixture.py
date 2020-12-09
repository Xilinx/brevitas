import pytest
import pytest_cases
from pytest_cases import fixture_union

from brevitas.core.bit_width import BitWidthConst, BitWidthParameter

from tests.brevitas.common import INT_BIT_WIDTH_TO_TEST, BOOLS


__all__ = [
    'bit_width',
    'min_bit_width',
    'override_pretrained',
    'bit_width_const',
    'bit_width_parameter',
    'bit_width_parameter_defaults',
    'bit_width_all'  # noqa
]


@pytest_cases.fixture()
@pytest_cases.parametrize('bit_width_init', INT_BIT_WIDTH_TO_TEST)
def bit_width(bit_width_init):
    """
    Integer bit-width value to initialize a bit-width module
    """
    return bit_width_init


@pytest_cases.fixture()
@pytest_cases.parametrize('min_bit_width_init', INT_BIT_WIDTH_TO_TEST)
def min_bit_width(min_bit_width_init):
    """
    Integer minimum bit-width value to initialize a bit-width module
    """
    return min_bit_width_init


@pytest_cases.fixture()
@pytest_cases.parametrize('override_pretrained_init', BOOLS)
def override_pretrained(override_pretrained_init):
    """
    Whether to ignore a pretrained bit_width that is loaded from state_dict or not
    """
    return override_pretrained_init


@pytest_cases.fixture()
def bit_width_const(bit_width):
    """
    Constant bit-width module
    """
    return BitWidthConst(bit_width)


@pytest_cases.fixture()
def bit_width_parameter_defaults(bit_width):
    """
    Learned bit-width with default arguments module
    """
    module = BitWidthParameter(bit_width)
    return module


@pytest_cases.fixture()
def bit_width_parameter(bit_width, min_bit_width, override_pretrained):
    """
    Learned bit-width with default arguments module
    """
    if bit_width < min_bit_width:
        pytest.xfail('bit_width cannot be smaller than min_bit_width')
    module = BitWidthParameter(
        bit_width,
        min_bit_width=min_bit_width,
        override_pretrained_bit_width=override_pretrained)
    return module


# Union of all variants of bit-width
fixture_union('bit_width_all', ['bit_width_const', 'bit_width_parameter'])