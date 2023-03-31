# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases

from brevitas.core.quant import TernaryQuant

__all__ = ['threshold_init', 'ternary_quant', 'delayed_ternary_quant']


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


@pytest_cases.fixture()
def delayed_ternary_quant(scaling_impl_all, quant_delay_steps, threshold_init):
    """
    Delayed TernaryQuant with all variants of scaling
    """
    return TernaryQuant(
        scaling_impl=scaling_impl_all,
        quant_delay_steps=quant_delay_steps,
        threshold=threshold_init)
