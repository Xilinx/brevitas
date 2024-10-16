# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas.core.function_wrapper.auto_round import AutoRoundSte
from brevitas.core.function_wrapper.learned_round import LearnedRoundHardSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
import brevitas.nn as qnn

OUT_CH = 16
IN_CH = 8
KERNEL_SIZE = 3
LEARNEDROUND_IMPL = [
    LearnedRoundSigmoid(),  # Sigmoid Implementation
    LearnedRoundSigmoid(learned_round_temperature=2.),  # Sigmoid + Temperature
    LearnedRoundHardSigmoid(),  # Hard Sigmoid
]


class TestLearnedRound():

    @pytest_cases.fixture()
    @pytest_cases.parametrize('impl', LEARNEDROUND_IMPL)
    def learnedround_float_to_int_impl(self, impl):
        sample_weight = torch.randn(OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE)
        impl = LearnedRoundSte(impl, torch.full(sample_weight.shape, 0.))

        # Simulate learned parameter
        impl.value.data = torch.randn_like(impl.value)
        return impl, sample_weight

    @pytest_cases.parametrize('training', [True, False])
    def test_learnedround(self, learnedround_float_to_int_impl, training):
        impl, sample_weight = learnedround_float_to_int_impl
        impl.train(training)

        out = impl(sample_weight)
        if training:
            # Soft quantization. All values are at most distant +/- 1 from the nearest integer
            assert torch.all(torch.abs(out - torch.round(out)) < 1)
        else:
            # Hard quantization. All values are integers
            assert torch.allclose(out, torch.round(out))

    def test_learnedround_load_dict(self, learnedround_float_to_int_impl):
        config.IGNORE_MISSING_KEYS = True

        impl, _ = learnedround_float_to_int_impl
        quant_conv = qnn.QuantConv2d(IN_CH, OUT_CH, KERNEL_SIZE, weight_float_to_int_impl=impl)
        fp_conv = torch.nn.Conv2d(IN_CH, OUT_CH, KERNEL_SIZE)
        try:
            quant_conv.load_state_dict(fp_conv.state_dict())
        except RuntimeError as e:
            pytest.fail(str(e))


class TestAutoRound():

    @pytest_cases.fixture()
    def autoround_float_to_int_impl(self):
        sample_weight = torch.randn(OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE)
        impl = AutoRoundSte(torch.full(sample_weight.shape, 0.))

        # Simulate learned parameter, values should be in the interval (-0.5, 0.5)
        impl.value.data = torch.rand_like(impl.value) * 0.5
        return impl, sample_weight

    def test_autoround(self, autoround_float_to_int_impl):
        impl, sample_weight = autoround_float_to_int_impl

        out = impl(sample_weight)
        # Check that all values are integers
        assert torch.allclose(out, torch.round(out))
        # Check that the values differ by at most 1 unit
        assert torch.all(torch.abs(sample_weight - out) < 1)

    def test_autoround_load_dict(self, autoround_float_to_int_impl):
        config.IGNORE_MISSING_KEYS = True

        impl, _ = autoround_float_to_int_impl
        quant_conv = qnn.QuantConv2d(IN_CH, OUT_CH, KERNEL_SIZE, weight_float_to_int_impl=impl)
        fp_conv = torch.nn.Conv2d(IN_CH, OUT_CH, KERNEL_SIZE)
        try:
            quant_conv.load_state_dict(fp_conv.state_dict())
        except RuntimeError as e:
            pytest.fail(str(e))

    def test_autoround_edge_cases(self):
        sample_weight = torch.tensor([-1.000, -0.500, 0.000, 0.500, 1.000])
        impl_data = torch.tensor([-0.500, 0.500, 0.000, -0.500, 0.500])
        impl = AutoRoundSte(impl_data)

        out = impl(sample_weight)
        # Check that all values are integers
        assert torch.allclose(out, torch.tensor([-2.000, 0.000, 0.000, 0.000, 2.000]))
