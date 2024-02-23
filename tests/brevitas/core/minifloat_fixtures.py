# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union

from brevitas.core.function_wrapper import FloatClamp
from brevitas.inject.enum import BitWidthImplType
from brevitas.quant.experimental.float_base import ExponentBiasMixin
from brevitas.quant.experimental.float_base import MaxFloatInfNaNMixin
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class Fp8e4m3Base(ExponentBiasMixin, MaxFloatInfNaNMixin, ScaledFloatWeightBase):
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3
    float_clamp_impl = FloatClamp
    nan_values = None
    inf_values = None
    bit_width_impl_type = BitWidthImplType.CONST
    # hypothesis extra
    hypothesis_internal_is_this_a_mock_check = False


class Fp8e5m2Base(ExponentBiasMixin, MaxFloatInfNaNMixin, ScaledFloatWeightBase):
    bit_width = 8
    exponent_bit_width = 5
    mantissa_bit_width = 2
    float_clamp_impl = FloatClamp
    nan_values = None
    inf_values = None
    bit_width_impl_type = BitWidthImplType.CONST
    # hypothesis extra
    hypothesis_internal_is_this_a_mock_check = False


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e4m3_regular(sat):

    class Fp8e4m3(Fp8e4m3Base):
        saturating = sat
        nan_values = tuple(('111',))
        inf_values = None

    return Fp8e4m3


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e5m2_regular(sat):

    class Fp8e5m2(Fp8e5m2Base):
        saturating = sat
        nan_values = ('01', '11', '10')
        inf_values = tuple(('00',))

    return Fp8e5m2


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e4m3_no_special_values(sat):

    class Fp8e4m3None(Fp8e4m3Base):
        saturating = sat

    return Fp8e4m3None


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e5m2_no_special_values(sat):

    class Fp8e5m2None(Fp8e5m2Base):
        saturating = sat

    return Fp8e5m2None


list_of_fixtures = [
    'fp8e4m3_regular', 'fp8e5m2_regular', 'fp8e4m3_no_special_values', 'fp8e5m2_no_special_values']

fp8_clamp = fixture_union('fp8_clamp', list_of_fixtures, ids=list_of_fixtures)
