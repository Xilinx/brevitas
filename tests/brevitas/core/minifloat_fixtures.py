# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union

from brevitas.core.function_wrapper import FloatClamp
from brevitas.quant.experimental.float_base import ExponentBiasMixin
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class Fp8e4m3Base(ExponentBiasMixin, ScaledFloatWeightBase):
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3
    case_clamp_impl = FloatClamp
    nan_values = tuple(('111',))
    inf_values = None
    # hypothesis extra
    hypothesis_internal_is_this_a_mock_check = False


class Fp8e5m2Base(ExponentBiasMixin, ScaledFloatWeightBase):
    bit_width = 8
    exponent_bit_width = 5
    mantissa_bit_width = 2
    case_clamp_impl = FloatClamp
    nan_values = ('01', '11', '10')
    inf_values = tuple(('00',))
    # hypothesis extra
    hypothesis_internal_is_this_a_mock_check = False


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e4m3(sat):

    class Fp8e4m3(Fp8e4m3Base):
        saturating = sat

    return Fp8e4m3


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e5m2(sat):

    class Fp8e5m2(Fp8e5m2Base):
        saturating = sat

    return Fp8e5m2


list_of_fixtures = ['fp8e4m3', 'fp8e5m2']

fp8_clamp = fixture_union('fp8_clamp', list_of_fixtures, ids=list_of_fixtures)
