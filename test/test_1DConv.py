import torch
import brevitas.nn.quant_conv1d as quant_conv1d
import test.generate_quant_input as input_gen
from brevitas.core.quant import QuantType

RAND_IT = 1
BIT = 8
SCALE = 0.2
ATOL = 1E-3
RTOL = 1E-5

BATCH = 10
IN_CHANNEL= 100
WIDTH = 21

OUT_CHANNEL = 200
KERNEL = 2
STRIDE = 2


class Test1DConv:

    def test_float_quant(self):
        shape = (BATCH, IN_CHANNEL, WIDTH)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        Conv1D = quant_conv1d.QuantConv1d(in_channels=IN_CHANNEL,
                                          out_channels=OUT_CHANNEL,
                                          kernel_size=KERNEL,
                                          stride=STRIDE,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False)

        results_float_quantized = Conv1D(input_quant)
        weight_int = Conv1D.int_weight
        bias = Conv1D.bias
        results_int_quantized = Conv1D.conv1d(input_quant_int, weight_int.float(), bias)
        totalScale = SCALE * Conv1D.quant_weight_scale
        result_rescaled = results_int_quantized * totalScale
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))

    def test_int(self):
        shape = (BATCH, IN_CHANNEL, WIDTH)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        Conv1D = quant_conv1d.QuantConv1d(in_channels=IN_CHANNEL,
                                          out_channels=OUT_CHANNEL,
                                          kernel_size=KERNEL,
                                          stride=STRIDE,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False)

        results_float_quantized = Conv1D(input_quant)
        weight_int = Conv1D.int_weight
        bias = Conv1D.bias
        results_int_quantized = Conv1D.conv1d(input_quant_int, weight_int.float(), bias)
        totalScale = SCALE * Conv1D.quant_weight_scale
        result_rescaled = torch.round(results_float_quantized / totalScale)
        assert (torch.allclose(results_int_quantized, result_rescaled, atol=ATOL, rtol=RTOL))

    def test_basic_padding(self):
        shape = (BATCH, IN_CHANNEL, WIDTH)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        Conv1D = quant_conv1d.QuantConv1d(in_channels=IN_CHANNEL,
                                          out_channels=OUT_CHANNEL,
                                          kernel_size=KERNEL,
                                          stride=STRIDE,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False,
                                          padding_type=quant_conv1d.PaddingType.SAME)

        results_float_quantized = Conv1D(input_quant)
        weight_int = Conv1D.int_weight
        bias = Conv1D.bias
        results_int_quantized = Conv1D.conv1d(input_quant_int, weight_int.float(), bias)
        totalScale = SCALE * Conv1D.quant_weight_scale
        result_rescaled = results_int_quantized * totalScale
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))


