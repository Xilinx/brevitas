import torch
import torch.nn.functional as F
import brevitas.nn.quant_conv1d as quant_conv1d
import test.generate_quant_input as input_gen
from brevitas.core.quant import QuantType

RAND_IT = 1
BIT = 8
SCALE = 0.2
ATOL = 1E-3
RTOL = 1E-5


class Test1DConv:

    def test_basic(self):
        shape = (10, 100, 21)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        Conv1D = quant_conv1d.QuantConv1d(in_channels=100,
                                          out_channels=200,
                                          kernel_size=2,
                                          stride=2,
                                          weight_quant_type=QuantType.INT,
                                          weight_bit_width=BIT,
                                          bias=False)

        results_float_quantized = Conv1D(input_quant)
        weight_int = Conv1D.int_weight
        bias = Conv1D.bias
        results_int_quantized = Conv1D.conv1d(input_quant_int, weight_int.float(), bias)
        totalScale = SCALE * Conv1D.quant_weight_scale
        result_rescaled = results_int_quantized * totalScale
        norm = torch.norm(results_float_quantized - result_rescaled)
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))

    def test_basic_padding(self):
        shape = (10, 100, 21)
        input_quant_int, input_quant = input_gen.generate_quant_input(shape, BIT, SCALE, True, True)
        Conv1D = quant_conv1d.QuantConv1d(in_channels=100,
                                          out_channels=200,
                                          kernel_size=2,
                                          stride=2,
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
        norm = torch.norm(results_float_quantized - result_rescaled)
        assert (torch.allclose(results_float_quantized, result_rescaled, atol= ATOL, rtol= RTOL))


