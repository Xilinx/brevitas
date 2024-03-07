# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union

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


list_of_fixtures = ['fp8e4m3', 'fp8e5m2']

fp8_clamp = fixture_union('fp8_clamp', list_of_fixtures, ids=list_of_fixtures)
