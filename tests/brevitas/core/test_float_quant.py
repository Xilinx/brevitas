# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import pytest
import torch

from brevitas.core.function_wrapper import FloatClamp
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling import ConstScaling
from brevitas.core.scaling import FloatScaling
from brevitas.function.ops import max_float
from brevitas.utils.torch_utils import float_internal_scale
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
                input_view_impl=Identity(),
                float_clamp_impl=None)
    else:
        # init FloatClamp
        float_clamp = FloatClamp(
            tensor_clamp_impl=TensorClamp(),
            signed=signed,
            inf_values=None,
            nan_values=None,
            saturating=True)
        float_scaling = FloatScaling(None, None, True)
        float_quant = FloatQuant(
            bit_width=bit_width,
            float_scaling_impl=float_scaling,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            input_view_impl=Identity(),
            signed=signed,
            float_clamp_impl=float_clamp)
        assert isinstance(float_quant.float_to_int_impl, RoundSte)
        assert isinstance(float_quant.float_scaling_impl, FloatScaling)
        assert isinstance(float_quant.scaling_impl, ConstScaling)


@given(minifloat_format=random_minifloat_format())
def test_minifloat(minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, _ = minifloat_format
    assert bit_width == exponent_bit_width + mantissa_bit_width + int(signed)


@given(inp=float_tensor_random_shape_st(), minifloat_format=random_minifloat_format())
@jit_disabled_for_mock()
def test_float_to_quant_float(inp, minifloat_format):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format

    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                input_view_impl=Identity(),
                signed=signed,
                float_clamp_impl=None)
    else:
        # init FloatClamp
        float_clamp = FloatClamp(
            tensor_clamp_impl=TensorClamp(),
            signed=signed,
            inf_values=None,
            nan_values=None,
            saturating=True)
        float_scaling_impl = mock.Mock(side_effect=lambda x, y, z: 1.)
        float_quant = FloatQuant(
            bit_width=bit_width,
            float_scaling_impl=float_scaling_impl,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            input_view_impl=Identity(),
            signed=signed,
            float_clamp_impl=float_clamp)
        expected_out, *_ = float_quant(inp)
        scale = float_quant.scaling_impl(inp)
        out_quant, scale = float_quant.quantize(inp, scale)
        exponent_bit_width, mantissa_bit_width, exponent_bias  = torch.tensor(exponent_bit_width, dtype=torch.float), torch.tensor(mantissa_bit_width, dtype=torch.float), torch.tensor(exponent_bias, dtype=torch.float)
        out_quant, *_ = float_quant.float_clamp_impl(
            out_quant, exponent_bit_width, mantissa_bit_width, exponent_bias)
        assert torch.allclose(expected_out, out_quant * scale)


@given(inp=float_tensor_random_shape_st(), minifloat_format=random_minifloat_format())
@jit_disabled_for_mock()
def test_scaling_impls_called_once(inp, minifloat_format):
    float_scaling_impl_return = 1.
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format
    scaling_impl = mock.Mock(side_effect=lambda x, y: 1.)
    float_scaling_impl = mock.Mock(side_effect=lambda x, y, z: float_scaling_impl_return)
    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                input_view_impl=Identity(),
                scaling_impl=scaling_impl,
                float_scaling_impl=float_scaling_impl,
                float_clamp_impl=None)
    else:
        # init FloatClamp
        float_clamp = FloatClamp(
            tensor_clamp_impl=TensorClamp(),
            signed=signed,
            inf_values=None,
            nan_values=None,
            saturating=True)
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            input_view_impl=Identity(),
            scaling_impl=scaling_impl,
            float_scaling_impl=float_scaling_impl,
            float_clamp_impl=float_clamp)
        float_scaling = float_scaling_impl(exponent_bit_width, mantissa_bit_width, exponent_bias)
        scale = float_quant.scaling_impl(inp, float_scaling)
        _ = float_quant.quantize(inp, scale)
        # scaling implementations should be called exaclty once on the input
        float_scaling_impl.assert_called_once_with(
            torch.tensor(exponent_bit_width),
            torch.tensor(mantissa_bit_width),
            torch.tensor(exponent_bias))
        scaling_impl.assert_called_once_with(inp, float_scaling_impl_return)


@given(
    inp=float_tensor_random_shape_st(),
    minifloat_format=random_minifloat_format(),
    scale=float_st())
@jit_disabled_for_mock()
def test_inner_scale(inp, minifloat_format, scale):
    bit_width, exponent_bit_width, mantissa_bit_width, signed, exponent_bias = minifloat_format
    # set scaling_impl to scale and float_scaling_impl to 1 to use the same scale as we are here
    float_scaling_impl = mock.Mock(side_effect=lambda x, y, z: 1.)
    scaling_impl = mock.Mock(side_effect=lambda x, y: scale)
    if exponent_bit_width == 0 or mantissa_bit_width == 0:
        with pytest.raises(RuntimeError):
            float_quant = FloatQuant(
                bit_width=bit_width,
                exponent_bit_width=exponent_bit_width,
                mantissa_bit_width=mantissa_bit_width,
                exponent_bias=exponent_bias,
                signed=signed,
                input_view_impl=Identity(),
                scaling_impl=scaling_impl,
                float_scaling_impl=float_scaling_impl,
                float_clamp_impl=None)
    else:
        # init FloatClamp
        float_clamp = FloatClamp(
            tensor_clamp_impl=TensorClamp(),
            signed=signed,
            inf_values=None,
            nan_values=None,
            saturating=True)
        float_quant = FloatQuant(
            bit_width=bit_width,
            exponent_bit_width=exponent_bit_width,
            mantissa_bit_width=mantissa_bit_width,
            exponent_bias=exponent_bias,
            signed=signed,
            input_view_impl=Identity(),
            scaling_impl=scaling_impl,
            float_scaling_impl=float_scaling_impl,
            float_clamp_impl=float_clamp)

        # scale inp manually
        scaled_inp = inp / scale
        max_val = max_float(
            torch.tensor(exponent_bit_width),
            torch.tensor(mantissa_bit_width),
            torch.tensor(exponent_bias))
        max_available_float = float_clamp.max_available_float
        max_value = max_val if max_available_float is None else torch.min(
            max_value, max_available_float)
        # call internal scale
        eps = torch.finfo(inp.dtype).tiny
        internal_scale = float_internal_scale(
            scaled_inp, float_quant.mantissa_bit_width(), float_quant.fp_internal_scale_min(), eps)
        val_fp_quant = internal_scale * float_quant.float_to_int_impl(scaled_inp / internal_scale)
        if signed:
            val_fp_quant = torch.clip(val_fp_quant, -1. * max_val, max_val)
        else:
            val_fp_quant = torch.clip(val_fp_quant, 0., max_val)

        # dequantize manually
        out = val_fp_quant * scale

        expected_out, expected_scale, *_ = float_quant(inp)

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
