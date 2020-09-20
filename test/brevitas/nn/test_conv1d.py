import torch
from brevitas.nn import QuantConv1d

from brevitas.core.quant import QuantType

from generate_quant_input import generate_quant_input
from common import check_expected_pyt_120_fail

# Quantization parameters
BIT = 8
SCALE = 0.2

# Absolute and Relative Tolerances
ATOL = 1E-3
RTOL = 1E-5

#Input Shape
BATCH = 10
IN_CHANNEL= 100
HEIGHT = 21

#Kernel/Output parameters
OUT_CHANNEL = 200
KERNEL = 2
STRIDE = 2


class Test1DConv:
    @check_expected_pyt_120_fail
    def test_float_quant(self):
        shape = (BATCH, IN_CHANNEL, HEIGHT)
        input_quant_int, input_quant = generate_quant_input(shape, BIT, SCALE, True, True)
        mod = QuantConv1d(
            in_channels=IN_CHANNEL,
            out_channels=OUT_CHANNEL,
            kernel_size=KERNEL,
            stride=STRIDE,
            weight_quant_type=QuantType.INT,
            weight_bit_width=BIT,
            bias=False)

        results_float_quantized = mod(input_quant)
        weight_int = mod.int_weight()
        bias = mod.bias
        results_int_quantized = mod.conv1d_zeros_pad(input_quant_int, weight_int.float(), bias)
        total_scale = SCALE * mod.quant_weight_scale()
        result_rescaled = results_int_quantized * total_scale
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))

    @check_expected_pyt_120_fail
    def test_int(self):
        shape = (BATCH, IN_CHANNEL, HEIGHT)
        input_quant_int, input_quant = generate_quant_input(shape, BIT, SCALE, True, True)
        mod = QuantConv1d(
            in_channels=IN_CHANNEL,
            out_channels=OUT_CHANNEL,
            kernel_size=KERNEL,
            stride=STRIDE,
            weight_quant_type=QuantType.INT,
            weight_bit_width=BIT,
            bias=False)

        results_float_quantized = mod(input_quant)
        weight_int = mod.int_weight()
        bias = mod.bias
        results_int_quantized = mod.conv1d_zeros_pad(input_quant_int, weight_int.float(), bias)
        total_scale = SCALE * mod.quant_weight_scale()
        result_rescaled = torch.round(results_float_quantized / total_scale)
        assert (torch.allclose(results_int_quantized, result_rescaled, atol=ATOL, rtol=RTOL))

    @check_expected_pyt_120_fail
    def test_basic_padding(self):
        shape = (BATCH, IN_CHANNEL, HEIGHT)
        input_quant_int, input_quant = generate_quant_input(shape, BIT, SCALE, True, True)
        mod = QuantConv1d(
            in_channels=IN_CHANNEL,
            out_channels=OUT_CHANNEL,
            kernel_size=KERNEL,
            stride=STRIDE,
            weight_quant_type=QuantType.INT,
            weight_bit_width=BIT,
            bias=False,
            padding_type='same')

        results_float_quantized = mod(input_quant)
        weight_int = mod.int_weight()
        bias = mod.bias
        results_int_quantized = mod.conv1d_same_zeros_pad(input_quant_int, weight_int.float(), bias)
        total_scale = SCALE * mod.quant_weight_scale()
        result_rescaled = results_int_quantized * total_scale
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))


