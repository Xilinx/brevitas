# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases
from pytest_cases import fixture_union

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.bit_width import BitWidthStatefulConst
from tests.brevitas.common import BOOLS
from tests.brevitas.common import INT_BIT_WIDTH_TO_TEST

__all__ = [
    'bit_width_init',
    'bit_width_init_two',
    'min_bit_width_init',
    'override_pretrained',
    'bit_width_const',
    'bit_width_stateful_const',
    'bit_width_parameter',
    'bit_width_parameter_defaults',
    'bit_width_all',  # noqa
    'bit_width_stateful'  # noqa
]


@pytest_cases.fixture()
@pytest_cases.parametrize('value', INT_BIT_WIDTH_TO_TEST)
def bit_width_init(value):
    """
    Integer bit-width value to initialize a bit-width module
    """
    return value


bit_width_init_two = bit_width_init


@pytest_cases.fixture()
@pytest_cases.parametrize('value', INT_BIT_WIDTH_TO_TEST)
def min_bit_width_init(value):
    """
    Integer minimum bit-width value to initialize a bit-width module
    """
    return value


@pytest_cases.fixture()
@pytest_cases.parametrize('override_pretrained_init', BOOLS)
def override_pretrained(override_pretrained_init):
    """
    Whether to ignore a pretrained bit_width that is loaded from state_dict or not
    """
    return override_pretrained_init


@pytest_cases.fixture()
def bit_width_const(bit_width_init):
    """
    Constant bit-width module
    """
    return BitWidthConst(bit_width_init)


# Specfying fixture scope to address macos pytest failures
@pytest_cases.fixture()
def bit_width_stateful_const(bit_width_init_two):
    """
    Constant bit-width module with persistent state
    """
    module = BitWidthStatefulConst(bit_width_init_two)
    return module


@pytest_cases.fixture()
def bit_width_parameter_defaults(bit_width_init):
    """
    Learned bit-width with default arguments module
    """
    module = BitWidthParameter(bit_width_init)
    return module


@pytest_cases.fixture()
def bit_width_parameter(bit_width_init, min_bit_width_init, override_pretrained):
    """
    Learned bit-width with default arguments module
    """
    if bit_width_init < min_bit_width_init:
        pytest.xfail('bit_width cannot be smaller than min_bit_width')
    module = BitWidthParameter(
        bit_width_init,
        min_bit_width=min_bit_width_init,
        override_pretrained_bit_width=override_pretrained)
    return module


# Union of all variants of bit-width
fixture_union(
    'bit_width_all', ['bit_width_const', 'bit_width_parameter', 'bit_width_stateful_const'])

# Union of all stateful variants of bit-width
fixture_union('bit_width_stateful', ['bit_width_parameter', 'bit_width_stateful_const'])
