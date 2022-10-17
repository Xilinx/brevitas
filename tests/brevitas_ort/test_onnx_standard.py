from operator import mul
from functools import reduce

import torch
import pytest

from brevitas.quant.scaled_int import Int32Bias
from .common import *

@pytest.mark.parametrize('export_type', ['qop', 'qcdq'])
@pytest.mark.parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('per_channel', [True, False])
@pytest.mark.parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
def test_standard_onnx_quant_conv(
        export_type, input_bit_width, weight_bit_width, output_bit_width, per_channel, quantizers):
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 1 # keep float error during fake-quantization under control
    weight_quant, io_quant = quantizers

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=weight_quant,
                input_quant=io_quant,
                output_quant=io_quant,
                weight_bit_width=weight_bit_width,
                input_bit_width=input_bit_width,
                output_bit_width=output_bit_width,
                weight_scaling_per_output_channel=per_channel,
                return_quant_tensor=True)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    export_name = 'qlinearconv.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=TOLERANCE)


@pytest.mark.parametrize('export_type', ['qop', 'qcdq'])
@pytest.mark.parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
@pytest.mark.parametrize('per_channel', [True, False])
@pytest.mark.parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
def test_standard_onnx_quant_linear_export(
        export_type, input_bit_width, weight_bit_width, output_bit_width, per_channel, quantizers):
    IN_SIZE = (1, IN_CH)
    weight_quant, io_quant = quantizers

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=True,
                bias_quant=Int32Bias,
                weight_quant=weight_quant,
                input_quant=io_quant,
                output_quant=io_quant,
                weight_bit_width=weight_bit_width,
                input_bit_width=input_bit_width,
                output_bit_width=output_bit_width,
                weight_scaling_per_output_channel=per_channel,
                return_quant_tensor=True)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    export_name = 'quantlinear_to_qlinearconv.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=TOLERANCE)

