# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import pytest
import torch

from brevitas.core.function_wrapper import RoundSte
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling import ConstScaling
from tests.brevitas.core.bit_width_fixture import *  # noqa
from tests.brevitas.core.int_quant_fixture import *  # noqa
from tests.brevitas.core.shared_quant_fixture import *  # noqa
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import random_minifloat_format
from tests.brevitas.hyp_helper import scalar_float_p_tensor_st
from tests.marker import jit_disabled_for_mock


@given(minifloat_format=random_minifloat_format())
def test_float_quant_defaults(minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed = minifloat_format
    # specifically don't set exponent bias to see if default works
    expected_exponent_bias = 2 ** (exponent_bit_width - 1) - 1
    float_quant = FloatQuant(
        bit_width=bit_width,
        signed=signed,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width)
    assert expected_exponent_bias == float_quant.exponent_bias()
    assert isinstance(float_quant.float_to_int_impl, RoundSte)
    assert isinstance(float_quant.float_scaling_impl, ConstScaling)
    assert isinstance(float_quant.scaling_impl, ConstScaling)


@given(minifloat_format=random_minifloat_format())
def test_minifloat(minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed = minifloat_format
    assert bit_width == exponent_bit_width + mantissa_bit_width + int(signed)


@given(inp=float_tensor_random_shape_st(), minifloat_format=random_minifloat_format())
@jit_disabled_for_mock()
def test_int_quant_to_in(inp, minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed = minifloat_format
    exponent_bias = 2 ** (exponent_bit_width - 1) - 1
    float_quant = FloatQuant(
        bit_width=bit_width,
        signed=signed,
        exponent_bit_width=exponent_bit_width,
        mantissa_bit_width=mantissa_bit_width,
        exponent_bias=exponent_bias)
    expected_out, _, _, bit_width_out = float_quant(inp)

    out_quant, scale = float_quant.quantize(inp)
    assert bit_width_out == bit_width
    assert torch.equal(expected_out, out_quant * scale)
