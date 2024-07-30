# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union

from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeight
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPWeight


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e4m3(sat):

    class Fp8e4m3(Fp8e4m3OCPWeight):
        saturating = sat
        # for hypothesis and DI
        hypothesis_internal_is_this_a_mock_check = True

    return Fp8e4m3


@pytest_cases.fixture
@pytest_cases.parametrize('sat', [True, False])
def fp8e5m2(sat):

    class Fp8e5m2(Fp8e5m2OCPWeight):
        saturating = sat
        # for hypothesis and DI
        hypothesis_internal_is_this_a_mock_check = True

    return Fp8e5m2


class Fp8CustomMixin(ExtendedInjector):
    bit_width = 8
    saturating = True

    hypothesis_internal_is_this_a_mock_check = True

    @value
    def mantissa_bit_width(bit_width, exponent_bit_width):
        return bit_width - exponent_bit_width - 1  # Sign bit


class Fp8e7m0Weight(Fp8CustomMixin, FloatWeightBase):
    exponent_bit_width = 7


class Fp8e6m1Weight(Fp8CustomMixin, FloatWeightBase):
    exponent_bit_width = 6


class Fp8e3m4Weight(Fp8CustomMixin, FloatWeightBase):
    exponent_bit_width = 3


class Fp8e2m5Weight(Fp8CustomMixin, FloatWeightBase):
    exponent_bit_width = 2


class Fp8e1m6Weight(Fp8CustomMixin, FloatWeightBase):
    exponent_bit_width = 1


@pytest_cases.fixture
@pytest_cases.parametrize('exponent_bit_width', [1, 2, 3, 6, 7])  # at least 1 exponent bit
def fp8Custom(exponent_bit_width):

    custom_exponents = {
        1: Fp8e1m6Weight,
        2: Fp8e2m5Weight,
        3: Fp8e3m4Weight,
        6: Fp8e6m1Weight,
        7: Fp8e7m0Weight,}

    return custom_exponents[exponent_bit_width]


list_of_fixtures = ['fp8e4m3', 'fp8e5m2', 'fp8Custom']

fp8_clamp = fixture_union('fp8_clamp', list_of_fixtures, ids=list_of_fixtures)
