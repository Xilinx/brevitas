# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
from hypothesis import strategies as st
import mock
import torch
from torch import Tensor

from brevitas.core.quant import TernaryQuant
from tests.brevitas.common import assert_allclose
from tests.brevitas.core.shared_quant_fixture import *  # noqa
from tests.brevitas.core.ternary_quant_fixture import *  # noqa
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import scalar_float_p_tensor_st
from tests.marker import jit_disabled_for_mock


def is_ternary_output_value_correct(scale: Tensor, output: Tensor):
    return ((output == scale) | (output == 0.0) | (output == -scale)).all()


def is_ternary_output_sign_correct(inp: Tensor, scale_thr: Tensor, output: Tensor):
    return (((output > torch.tensor(0.0)) & (inp > scale_thr)) |
            ((output == torch.tensor(0.0)) & (inp >= -scale_thr) &
             (inp <= scale_thr)) | ((output < torch.tensor(0.0)) & (inp < -scale_thr))).all()


class TestTernaryUnit:

    @given(
        inp=float_tensor_random_shape_st(),
        scale_init=scalar_float_p_tensor_st(),
        threshold=st.floats(min_value=0.0, max_value=1.0))
    @jit_disabled_for_mock()
    def test_ternary_quant(self, inp, scale_init, threshold):
        scaling_impl = mock.Mock(return_value=scale_init)
        ternary_quant = TernaryQuant(scaling_impl, threshold)
        output, scale, zp, bit_width = ternary_quant(inp)
        scaling_impl.assert_called_once_with(inp)
        assert is_ternary_output_value_correct(scale, output)
        assert is_ternary_output_sign_correct(inp, scale * threshold, output)
        assert (scale == scale_init).all()
        assert zp == torch.tensor(0.0)
        assert bit_width == torch.tensor(2.0)


class TestTernaryIntegration:

    def test_assigned_bit_width(self, ternary_quant):
        assert ternary_quant.bit_width() == torch.tensor(2.0)

    def test_assigned_zero_point(self, ternary_quant):
        assert ternary_quant.zero_point() == torch.tensor(0.0)

    def test_assigned_threshold(self, ternary_quant, threshold_init):
        assert ternary_quant.threshold == threshold_init

    @given(inp=float_tensor_random_shape_st())
    def test_output_sign(self, ternary_quant, inp):
        output, scale, _, _ = ternary_quant(inp)
        scale_thr = ternary_quant.threshold * scale
        assert is_ternary_output_sign_correct(inp, scale_thr, output)

    @given(inp=float_tensor_random_shape_st())
    def test_output_value(self, ternary_quant, inp):
        output, scale, _, _ = ternary_quant(inp)
        assert is_ternary_output_value_correct(scale, output)

    def test_delayed_output_value(self, delayed_ternary_quant, quant_delay_steps, randn_inp):
        """
        Test delayed quantization by a certain number of steps. Because delayed quantization is
        stateful, we can't use Hypothesis to generate the input, so we resort to a basic fixture.
        """
        for i in range(quant_delay_steps):
            output, _, _, _ = delayed_ternary_quant(randn_inp)
            assert (output == randn_inp).all()
        output, scale, _, _ = delayed_ternary_quant(randn_inp)
        assert is_ternary_output_value_correct(scale, output)

    @given(inp=float_tensor_random_shape_st())
    def test_output_bit_width(self, ternary_quant, inp):
        _, _, _, bit_width = ternary_quant(inp)
        assert_allclose(bit_width, torch.tensor(2.0))

    @given(inp=float_tensor_random_shape_st())
    def test_output_zero_point(self, ternary_quant, inp):
        _, _, zero_point, _ = ternary_quant(inp)
        assert_allclose(zero_point, torch.tensor(0.0))

    @given(inp=float_tensor_random_shape_st())
    def test_output_scale(self, ternary_quant, scaling_impl_all, inp):
        _, scale, _, _ = ternary_quant(inp)
        assert_allclose(scale, scaling_impl_all(inp))
