import pytest
import torch

from brevitas.nn import QuantReLU, QuantIdentity


class TestQuantReLU:

    def test_module_init_default(self):
        mod = QuantReLU(max_val=6)

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