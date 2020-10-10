from torch.nn import Module, Conv2d, BatchNorm2d

from brevitas.nn import QuantConv2d
from brevitas.inject.defaults import Int8BiasPerTensorFloatInternalScaling
from brevitas.nn.utils import merge_bn

import torch

OUTPUT_CHANNELS = 10
INPUT_CHANNELS = 5
KERNEL_SIZE = (3, 3)
WEIGHT_BIT_WIDTH = 5


class TestQuantConv2d:

    def test_module_init(self):
        mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)

    def test_fp_quant_module(self):
        float_mod = Conv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_type='FP',
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_none_weight_quant_module(self):
        float_mod = Conv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant=None,
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_delayed_quant_module(self):
        float_mod = Conv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)
        quant_mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=False)
        quant_mod.load_state_dict(float_mod.state_dict())
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20)
        out_float = float_mod(inp)
        out_quant = quant_mod(inp)
        assert out_float.isclose(out_quant).all().item()

    def test_internally_scaled_int_bias(self):
        mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=True,
            bias_quant=Int8BiasPerTensorFloatInternalScaling)
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20)
        mod(inp)

    def test_internally_scaled_int_bias_after_bn_merge(self):
        mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            weight_quant_delay_steps=1,
            bias=False,
            bias_quant=Int8BiasPerTensorFloatInternalScaling)
        bn = BatchNorm2d(OUTPUT_CHANNELS)
        merge_bn(mod, bn)
        inp = torch.randn(1, INPUT_CHANNELS, 20, 20)
        mod(inp)