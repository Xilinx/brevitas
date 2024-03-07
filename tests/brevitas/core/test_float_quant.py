# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import pytest
import torch

from brevitas.core.function_wrapper import FloatClamp
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling import ConstScaling
from brevitas.utils.float_quant_utils import get_max_value
from tests.brevitas.hyp_helper import float_st
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import random_minifloat_format
from tests.marker import jit_disabled_for_mock


@given(minifloat_format=random_minifloat_format())
def test_float_quant_defaults(minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format

    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                float_clamp_impl=None)
    else:
        max_value = get_max_value(
            exponent_bit_width, mantissa_bit_width, exponent_bias, None, None, True)
        # init FloatClamp
        float_clamp = FloatClamp(max_value=max_value, tensor_clamp_impl=TensorClamp())
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            float_clamp_impl=float_clamp)
        assert isinstance(float_quant.float_to_int_impl, RoundSte)
        assert isinstance(float_quant.float_scaling_impl, ConstScaling)
        assert isinstance(float_quant.scaling_impl, ConstScaling)


@given(minifloat_format=random_minifloat_format())
def test_minifloat(minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, _ = minifloat_format
    assert bit_width == exponent_bit_width + mantissa_bit_width + int(signed)


@given(inp=float_tensor_random_shape_st(), minifloat_format=random_minifloat_format())
def test_float_to_quant_float(inp, minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format
    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                float_clamp_impl=None)
    else:
        max_value = get_max_value(
            exponent_bit_width, mantissa_bit_width, exponent_bias, None, None, True)
        # init FloatClamp
        float_clamp = FloatClamp(max_value=max_value, tensor_clamp_impl=TensorClamp())
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            float_clamp_impl=float_clamp)
        expected_out, _, _, bit_width_out = float_quant(inp)

        out_quant, scale = float_quant.quantize(inp)
        assert bit_width_out == bit_width
        assert torch.equal(expected_out, out_quant * scale)


@given(inp=float_tensor_random_shape_st(), minifloat_format=random_minifloat_format())
@jit_disabled_for_mock()
def test_scaling_impls_called_once(inp, minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format
    scaling_impl = mock.Mock(side_effect=lambda x: 1.)
    float_scaling_impl = mock.Mock(side_effect=lambda x: 1.)
    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                scaling_impl=scaling_impl,
                float_scaling_impl=float_scaling_impl,
                float_clamp_impl=None)
    else:
        max_value = get_max_value(
            exponent_bit_width, mantissa_bit_width, exponent_bias, None, None, True)
        # init FloatClamp
        float_clamp = FloatClamp(max_value=max_value, tensor_clamp_impl=TensorClamp())
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            scaling_impl=scaling_impl,
            float_scaling_impl=float_scaling_impl,
            float_clamp_impl=float_clamp)
        output = float_quant.quantize(inp)
        # scaling implementations should be called exaclty once on the input
        scaling_impl.assert_called_once_with(inp)
        float_scaling_impl.assert_called_once_with(inp)


@given(
    inp=float_tensor_random_shape_st(),
    minifloat_format=random_minifloat_format(),
    scale=float_st())
@jit_disabled_for_mock()
def test_inner_scale(inp, minifloat_format, scale):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format
    # set scaling_impl to scale and float_scaling_impl to 1 to use the same scale as we are here
    scaling_impl = mock.Mock(side_effect=lambda x: scale)
    float_scaling_impl = mock.Mock(side_effect=lambda x: 1.)
    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                scaling_impl=scaling_impl,
                float_scaling_impl=float_scaling_impl,
                float_clamp_impl=None)
    else:
        max_value = get_max_value(
            exponent_bit_width, mantissa_bit_width, exponent_bias, None, None, True)
        # init FloatClamp
        float_clamp = FloatClamp(max_value=max_value, tensor_clamp_impl=TensorClamp())
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            scaling_impl=scaling_impl,
            float_scaling_impl=float_scaling_impl,
            float_clamp_impl=float_clamp)

        # scale inp manually
        scaled_inp = inp / scale

        # call internal scale
        internal_scale = float_quant.internal_scale(scaled_inp)
        val_fp_quant = internal_scale * float_quant.float_to_int_impl(scaled_inp / internal_scale)
        if signed:
            val_fp_quant = torch.clip(
                val_fp_quant, -1. * float_quant.fp_max_val(), float_quant.fp_max_val())
        else:
            val_fp_quant = torch.clip(val_fp_quant, 0., float_quant.fp_max_val())

        # dequantize manually
        out = val_fp_quant * scale

        expected_out, expected_scale, _, _ = float_quant(inp)

        assert scale == expected_scale
        if scale == 0.0:
            # outputs should only receive 0s or nan
            assert torch.tensor([
                True if val == 0. or val.isnan() else False for val in out.flatten()]).all()
            assert torch.tensor([
                True if val == 0. or val.isnan() else False for val in expected_out.flatten()
            ]).all()
        else:
            # filter out NaN values as we can't compare them
            # Note: this still checks if NaN appears at the same values
            out_nans = out.isnan()
            expected_out_nans = expected_out.isnan()
            assert torch.equal(out[~out_nans], expected_out[~expected_out_nans])
