import numpy as np
import pytest
from pytest_cases import parametrize
from torch import nn

from brevitas.quant.scaled_int import Int32Bias
from brevitas_ort.common import *


class QuantWBIOLCases:

    @parametrize('impl', QUANT_WBIOL_IMPL)
    @parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
    @parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
    @parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
    @parametrize('per_channel', [True, False])
    @parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
    def case_quant_wbiol(
            self, impl, input_bit_width, weight_bit_width, output_bit_width, per_channel, quantizers):
        weight_quant, io_quant = quantizers
        if impl is QuantLinear:
            layer_kwargs = {
                'in_features': IN_CH,
                'out_features': OUT_CH}
        else:
            layer_kwargs = {
                'in_channels': IN_CH,
                'out_channels': OUT_CH,
                'kernel_size': KERNEL_SIZE}

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = impl(
                    **layer_kwargs,
                    bias=True,
                    weight_quant=weight_quant,
                    input_quant=io_quant,
                    output_quant=io_quant,
                    weight_bit_width=weight_bit_width,
                    input_bit_width=input_bit_width,
                    output_bit_width=output_bit_width,
                    bias_quant=Int32Bias,
                    weight_scaling_per_output_channel=per_channel,
                    return_quant_tensor=True)
                self.conv.weight.data.uniform_(-0.01, 0.01)

            def forward(self, x):
                return self.conv(x)

        module = Model()
        return module


