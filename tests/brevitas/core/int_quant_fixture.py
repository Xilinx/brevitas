# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases
import torch

from brevitas.core.quant import IntQuant
from tests.brevitas.common import BOOLS

__all__ = ['int_quant', 'signed', 'narrow_range', 'zero_point_init', 'arange_int_tensor']


@pytest_cases.fixture()
@pytest_cases.parametrize('value', BOOLS)
def signed(value):
    """
    Is quantization signed or not.
    """
    return value


@pytest_cases.fixture()
@pytest_cases.parametrize('value', BOOLS)
def narrow_range(value):
    """
    Is quantization narrow-range or not.
    """
    return value


@pytest_cases.fixture()
@pytest_cases.parametrize('multiplier', [0, 0.3, 0.7])
def zero_point_init(bit_width_init, multiplier):
    """
    Value to initialize zero-point with, based on bit-width
    """
    return int(round(multiplier * (2 ** (bit_width_init - 1) - 1)))


@pytest_cases.fixture()
def arange_int_tensor(signed, narrow_range, bit_width_init):
    """
    Generate the range of integers covered by the bit_width/signed/narrow_range parametrization
    """
    if not signed and not narrow_range:
        t = torch.arange(0, (2 ** bit_width_init) - 1 + 1)
    elif not signed and narrow_range:
        t = torch.arange(0, (2 ** bit_width_init) - 2 + 1)
    elif signed and not narrow_range:
        t = torch.arange(-(2 ** (bit_width_init - 1)), (2 ** (bit_width_init - 1) - 1) + 1)
    else:
        t = torch.arange(-((2 ** (bit_width_init - 1)) - 1), (2 ** (bit_width_init - 1) - 1) + 1)
    return t


@pytest_cases.fixture()
def int_quant(float_to_int_impl, tensor_clamp_impl, signed, narrow_range):
    """
    Int quant with external bit-width, scale, zero-point.
    """
    return IntQuant(
        float_to_int_impl=float_to_int_impl,
        tensor_clamp_impl=tensor_clamp_impl,
        signed=signed,
        narrow_range=narrow_range)
