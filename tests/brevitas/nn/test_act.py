# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.nn import QuantHardSwish
from brevitas.nn import QuantHardTanh
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantReLU


class TestQuantHardTanh:

    def test_module_init_min_max(self):
        mod = QuantHardTanh(min_val=-1.0, max_val=1.0)


class TestQuantReLU:

    def test_module_init_default(self):
        mod = QuantReLU()

    def test_module_init_const_scaling(self):
        mod = QuantReLU(max_val=6, scaling_impl_type='CONST')


class TestQuantHardSwish:

    def test_module_init_default(self):
        mod = QuantHardSwish()

    def test_module_init_with_bit_width(self):
        # Test with custom bit width
        mod = QuantHardSwish(bit_width=4)

    def test_forward_pass(self):
        mod = QuantHardSwish()
        inp = torch.randn(1, 20, 10, 10)
        out = mod(inp)
        assert out.shape == inp.shape

    def test_output_behavior(self):
        mod = QuantHardSwish()
        mod.eval()
        # For large positive inputs, hardswish output should be close to input (positive)
        inp_positive = torch.tensor([5.0, 10.0, 100.0])
        out_positive = mod(inp_positive)
        assert (out_positive >= 0).all().item()

        # For small negative inputs, unquantized hardswish can produce small negative values
        # HardSwish(x) ≈ -0.33 at minimum (around x=-3)
        # With unsigned quantization (default), these small negative values are clamped to zero
        inp_negative = torch.tensor([-2.0, -1.0, -0.5])
        out_negative = mod(inp_negative)
        assert (out_negative >= 0).all().item()

    def test_training_eval_modes(self):
        mod = QuantHardSwish()
        inp = torch.randn(2, 6, 16, 16)

        # Training mode
        mod.train()
        out_train = mod(inp)
        assert out_train.shape == inp.shape

        # Eval mode
        mod.eval()
        out_eval = mod(inp)
        assert out_eval.shape == inp.shape


class TestQuantDelay:

    @pytest.mark.parametrize("bw_quant_type", [(4, "INT"), (1, "BINARY"), (2, "TERNARY")])
    def test_quant_identity_delay(self, bw_quant_type):
        DELAY = 10
        bit_width, quant_type = bw_quant_type
        mod = QuantIdentity(
            min_val=-6.0,
            max_val=6.0,
            threshold=0.5,  # for ternary quant
            bit_width=bit_width,
            quant_type=quant_type,
            quant_delay_steps=DELAY)
        for i in range(DELAY):
            t = torch.randn(1, 10, 5, 5)
            out = mod(t)
            assert t.isclose(out).all().item()
        t = torch.randn(1, 10, 5, 5)
        out = mod(t)
        assert not t.isclose(out).all().item()
