# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

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
