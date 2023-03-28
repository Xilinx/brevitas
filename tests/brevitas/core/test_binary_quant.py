# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import torch
from torch import Tensor

from tests.brevitas.common import assert_allclose
from tests.brevitas.core.binary_quant_fixture import *  # noqa
from tests.brevitas.core.shared_quant_fixture import *  # noqa
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import scalar_float_p_tensor_st
from tests.marker import jit_disabled_for_mock


def is_binary_output_value_correct(scale: Tensor, output: Tensor):
    return ((output == scale) | (output == -scale)).all()


def is_binary_output_sign_correct(inp: Tensor, output: Tensor):
    return (((output > torch.tensor(0.0)) & (inp >= torch.tensor(0))) |
            ((output < torch.tensor(0.0)) & (inp < torch.tensor(0)))).all()


class TestBinaryUnit:

    @given(inp=float_tensor_random_shape_st(), scale_init=scalar_float_p_tensor_st())
    @jit_disabled_for_mock()
    def test_binary_quant(self, binary_quant_impl_all, inp, scale_init):
        scaling_impl = mock.Mock(return_value=scale_init)
        binary_quant = binary_quant_impl_all(scaling_impl)
        output, scale, zp, bit_width = binary_quant(inp)
        scaling_impl.assert_called_once_with(inp)
        assert is_binary_output_value_correct(scale, output)
        assert is_binary_output_sign_correct(inp, output)
        assert (scale == scale_init).all()
        assert zp == torch.tensor(0.0)
        assert bit_width == torch.tensor(1.0)


class TestBinaryIntegration:

    def test_assigned_bit_width(self, binary_quant_all):
        assert binary_quant_all.bit_width() == torch.tensor(1.0)

    def test_assigned_zero_point(self, binary_quant_all):
        assert binary_quant_all.zero_point() == torch.tensor(0.0)

    @given(inp=float_tensor_random_shape_st())
    def test_output_sign(self, binary_quant_all, inp):
        output, _, _, _ = binary_quant_all(inp)
        assert is_binary_output_sign_correct(inp, output)

    @given(inp=float_tensor_random_shape_st())
    def test_output_value(self, binary_quant_all, inp):
        output, scale, _, _ = binary_quant_all(inp)
        assert is_binary_output_value_correct(scale, output)

    def test_delayed_output_value(self, delayed_binary_quant_all, quant_delay_steps, randn_inp):
        """
        Test delayed quantization by a certain number of steps. Because delayed quantization is
        stateful, we can't use Hypothesis to generate the input, so we resort to a basic fixture.
        """
        for i in range(quant_delay_steps):
            output, _, _, _ = delayed_binary_quant_all(randn_inp)
            assert (output == randn_inp).all()
        output, scale, _, _ = delayed_binary_quant_all(randn_inp)
        assert is_binary_output_value_correct(scale, output)

    @given(inp=float_tensor_random_shape_st())
    def test_output_bit_width(self, binary_quant_all, inp):
        _, _, _, bit_width = binary_quant_all(inp)
        assert_allclose(bit_width, torch.tensor(1.0))

    @given(inp=float_tensor_random_shape_st())
    def test_output_zero_point(self, binary_quant_all, inp):
        _, _, zero_point, _ = binary_quant_all(inp)
        assert_allclose(zero_point, torch.tensor(0.0))

    @given(inp=float_tensor_random_shape_st())
    def test_output_scale(self, binary_quant_all, scaling_impl_all, inp):
        _, scale, _, _ = binary_quant_all(inp)
        assert_allclose(scale, scaling_impl_all(inp))
