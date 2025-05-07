# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
from hypothesis.strategies import floats
import pytest
import pytest_cases
import torch
import torch.nn as nn

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundHardSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundIdentity
from brevitas.core.function_wrapper.learned_round import LearnedRoundSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
import brevitas.nn as qnn
from tests.brevitas.hyp_helper import two_float_tensor_random_shape_st

OUT_CH = 16
IN_CH = 8
KERNEL_SIZE = 3
LEARNEDROUND_IMPL = [
    LearnedRoundSigmoid(),  # Sigmoid Implementation
    LearnedRoundSigmoid(learned_round_temperature=2.),  # Sigmoid + Temperature
    LearnedRoundHardSigmoid(),  # Hard Sigmoid
    LearnedRoundIdentity(),]


class TestLearnedRound():

    def instantiate_learnedround_float_to_int_impl(self, impl, weights, value):
        impl = LearnedRoundSte(impl, torch.full(weights.shape, 0.), nn.Identity())
        if isinstance(impl.learned_round_impl, LearnedRoundIdentity):
            min_value, max_value = torch.min(value), torch.max(value)
            # Prevent division by zero when all the elements of the tensor are the same
            if max_value - min_value < 1e-8:
                # Make sure that the division is safe
                if torch.abs(max_value) > 1e-8:
                    value = value / max_value - 0.5
            else:
                value = (value - min_value) / (max_value - min_value) - 0.5
        # Simulate learned round
        impl.value.data = value
        return impl

    # NOTE: The min/max values are set to the exactly representable float32
    # closer to sys.maxsize for a given machine.
    @pytest_cases.parametrize('impl', LEARNEDROUND_IMPL)
    @pytest_cases.parametrize('training', [True, False])
    @given(weights_value=two_float_tensor_random_shape_st(min_val=-(2 ** 8), max_val=2 ** 8))
    def test_learnedround(self, impl, training, weights_value):
        # Unpack tuple of hypothesis generated tensors
        weights, value = weights_value
        # Instantiate LearnedRoundSte using fabric method
        impl = self.instantiate_learnedround_float_to_int_impl(impl, weights, value)
        impl.train(training)
        print(impl.value)
        out = impl(weights)
        # The FP values and its quantized values must differ by at most +/- 1
        assert torch.all(torch.abs(out - weights) <= 1)
        if not isinstance(impl.learned_round_impl, LearnedRoundIdentity):
            if training:
                # Soft quantization. All values are at most distant +/- 1 from the nearest integer
                assert torch.all(torch.abs(out - torch.round(out)) <= 1)
            else:
                # Hard quantization. All values are integers
                assert torch.allclose(out, torch.round(out))
        else:
            # All values should be integers for LearnedRoundIdentity
            assert torch.allclose(out, torch.round(out))

    @given(
        learned_round_zeta=floats(min_value=0.0, max_value=3.0),
        learned_round_gamma=floats(min_value=-3.0, max_value=-0.05),
        value=floats(min_value=-5.0, max_value=5.0),
    )
    def test_learnedround_float_to_int_impl_hard_sigmoid(
            self, learned_round_zeta, learned_round_gamma, value):
        value = torch.tensor([value], dtype=torch.float32)
        weight = torch.zeros_like(value)
        # Initialise learned round script module
        learned_round_hard_sigmoid = LearnedRoundHardSigmoid(
            learned_round_zeta=learned_round_zeta,
            learned_round_gamma=learned_round_gamma,
        )
        learned_round_hard_sigmoid.train(False)
        value_eval = learned_round_hard_sigmoid(value)
        learned_round_hard_sigmoid.train(True)
        value_train = learned_round_hard_sigmoid(value)

        out_eval = weight + value_eval
        out_train = weight + (value_train > 0.5)

        assert torch.allclose(out_eval, out_train)

    @pytest_cases.fixture()
    @pytest_cases.parametrize('impl', LEARNEDROUND_IMPL)
    def learnedround_float_to_int_impl(self, impl):
        sample_weight = torch.randn(OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE)
        impl = LearnedRoundSte(impl, torch.full(sample_weight.shape, 0.), nn.Identity())

        # Simulate learned parameter
        value = torch.randn_like(impl.value)
        impl.value.data = value
        return impl, sample_weight, value

    def test_learnedround_load_dict(self, learnedround_float_to_int_impl):
        config.IGNORE_MISSING_KEYS = True

        impl, _, _ = learnedround_float_to_int_impl
        quant_conv = qnn.QuantConv2d(IN_CH, OUT_CH, KERNEL_SIZE, weight_float_to_int_impl=impl)
        fp_conv = torch.nn.Conv2d(IN_CH, OUT_CH, KERNEL_SIZE)
        try:
            quant_conv.load_state_dict(fp_conv.state_dict())
        except RuntimeError as e:
            pytest.fail(str(e))

    def test_learnedround_state_dict(self, learnedround_float_to_int_impl):
        impl, _, value = learnedround_float_to_int_impl
        state_dict = impl.state_dict()

        # Verify that the state dict contains the entry corresponding to the
        # learnable round parameter.
        assert len(state_dict.keys()) == 1
        assert torch.allclose(state_dict["value"], value)
