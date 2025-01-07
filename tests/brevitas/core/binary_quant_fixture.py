# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union

from brevitas.core.quant import BinaryQuant
from brevitas.core.quant import ClampedBinaryQuant

__all__ = [
    'binary_quant',
    'clamped_binary_quant',
    'binary_quant_impl_all',
    'binary_quant_all',  # noqa
]


@pytest_cases.fixture()
@pytest_cases.parametrize('impl', [BinaryQuant, ClampedBinaryQuant])
def binary_quant_impl_all(impl):
    """
    Classes implementing variants of binary quantization
    """
    return impl


@pytest_cases.fixture()
def binary_quant(scaling_impl_all):
    """
    Binary quant with all variants of scaling
    """
    return BinaryQuant(scaling_impl=scaling_impl_all)


@pytest_cases.fixture()
def clamped_binary_quant(scaling_impl_all):
    """
    ClampedBinaryQuant with all variants of scaling
    """
    return ClampedBinaryQuant(scaling_impl=scaling_impl_all)


fixture_union('binary_quant_all', ['binary_quant', 'clamped_binary_quant'])
