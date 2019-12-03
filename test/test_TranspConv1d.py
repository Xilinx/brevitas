import torch
import brevitas.nn.quant_ConvTranspose1d as quant_ConvTranspose1d
import test.generate_quant_input as input_gen
from brevitas.core.quant import QuantType
import random
import numpy as np

# Quantization parameters
BIT = 8
SCALE = 0.2

# Absolute and Relative Tolerances
ATOL = 1E-3
RTOL = 1E-5

#Input Shape
BATCH = 10
IN_CHANNEL= 100
HEIGHT = 500

#Kernel/Output parameters
OUT_CHANNEL = 200
KERNEL = 2
STRIDE = 2


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class Test1DTranspConv:

    def test_float_quant(self):
        shape = (BATCH, IN_CHANNEL, HEIGHT)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        ConvTranspose1d = quant_ConvTranspose1d.QuantConvTranspose1d(in_channels=IN_CHANNEL,
                                          out_channels=OUT_CHANNEL,
                                          kernel_size=KERNEL,
                                          stride=STRIDE,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False)

        results_float_quantized = ConvTranspose1d(input_quant)
        weight_int = ConvTranspose1d.int_weight
        bias = ConvTranspose1d.bias
        output_padding = ConvTranspose1d.compute_output_padding(input_quant_int, None)
        results_int_quantized = ConvTranspose1d.conv_transpose1d(input_quant_int, weight_int.float(), bias, output_padding)
        totalScale = SCALE * ConvTranspose1d.quant_weight_scale
        result_rescaled = results_int_quantized * totalScale
        # print(torch.norm(results_float_quantized- result_rescaled))
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))

    def test_int(self):
        shape = (BATCH, IN_CHANNEL, HEIGHT)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        ConvTranspose1d = quant_ConvTranspose1d.QuantConvTranspose1d(in_channels=IN_CHANNEL,
                                          out_channels=OUT_CHANNEL,
                                          kernel_size=KERNEL,
                                          stride=STRIDE,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False)

        results_float_quantized = ConvTranspose1d(input_quant)
        weight_int = ConvTranspose1d.int_weight
        bias = ConvTranspose1d.bias
        output_padding = ConvTranspose1d.compute_output_padding(input_quant_int, None)
        results_int_quantized = ConvTranspose1d.conv_transpose1d(input_quant_int, weight_int.float(), bias, output_padding)
        totalScale = SCALE * ConvTranspose1d.quant_weight_scale
        result_rescaled = torch.round(results_float_quantized / totalScale)
        assert (torch.allclose(results_int_quantized, result_rescaled, atol=ATOL, rtol=RTOL))

    # def test_int_scale_layer(self):
    #     shape = (BATCH, IN_CHANNEL, HEIGHT)
    #     input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
    #     ConvTranspose1d = quant_ConvTranspose1d.QuantConvTranspose1d(in_channels=IN_CHANNEL,
    #                                       out_channels=OUT_CHANNEL,
    #                                       kernel_size=KERNEL,
    #                                       stride=STRIDE,
    #                                       weight_quant_type=QuantType.INT,
    #                                       weight_bit_width=BIT,
    #                                       weight_scaling_per_output_channel=True,
    #                                       bias=False)
    #
    #     results_float_quantized = ConvTranspose1d(input_quant)
    #     weight_int = ConvTranspose1d.int_weight
    #     bias = ConvTranspose1d.bias
    #     output_padding = ConvTranspose1d.compute_output_padding(input_quant_int, None)
    #     results_int_quantized = ConvTranspose1d.conv_transpose1d(input_quant_int, weight_int.float(), bias, output_padding)
    #     totalScale = SCALE * ConvTranspose1d.quant_weight_scale
    #     result_rescaled = torch.round(results_float_quantized / totalScale)
    #     assert (torch.allclose(results_int_quantized, result_rescaled, atol=ATOL, rtol=RTOL))
