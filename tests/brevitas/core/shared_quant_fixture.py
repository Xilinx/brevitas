# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
from pytest_cases import fixture_union
import torch

from brevitas.core.scaling import ConstScaling
from brevitas.core.scaling import ParameterScaling

__all__ = [
    'quant_delay_steps',
    'const_scaling_impl',
    'parameter_scaling_impl',
    'standalone_scaling_init',
    'randn_inp',
    'scaling_impl_all'  # noqa
]


@pytest_cases.fixture()
@pytest_cases.parametrize('steps', [1, 10])
def quant_delay_steps(steps):
    """
    Non-zero steps to delay quantization
    """
    return steps


@pytest_cases.fixture()
def const_scaling_impl(standalone_scaling_init):
    """
    Scaling by a const implementation
    """
    return ConstScaling(standalone_scaling_init)


@pytest_cases.fixture()
def parameter_scaling_impl(standalone_scaling_init):
    """
    Scaling by a parameter implementation
    """
    return ParameterScaling(standalone_scaling_init)


@pytest_cases.fixture()
@pytest_cases.parametrize('value', [0.001, 5.0])
def standalone_scaling_init(value):
    """
    Value to initialize const/parameter scaling with
    """
    return value


@pytest_cases.fixture()
def randn_inp():
    """
    4-dim randn tensor
    """
    return torch.randn(size=(5, 4, 3, 2))


fixture_union('scaling_impl_all', ['const_scaling_impl', 'parameter_scaling_impl'])
