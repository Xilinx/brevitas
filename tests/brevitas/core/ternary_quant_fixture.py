# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases

from brevitas.core.quant import TernaryQuant

__all__ = ['threshold_init', 'ternary_quant']


@pytest_cases.fixture()
def threshold_init():
    """
    Threshold value for ternary quantization
    """
    return 0.5


@pytest_cases.fixture()
def ternary_quant(scaling_impl_all, threshold_init):
    """
    Ternary quant with all variants of scaling
    """
    return TernaryQuant(scaling_impl=scaling_impl_all, threshold=threshold_init)
