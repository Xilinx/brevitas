from pytest_cases import parametrize, set_case_id
from torch import nn
import torch

from ...conftest import SEED

from brevitas.nn import QuantLinear, QuantConv1d, QuantConv2d
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias, Int8BiasPerTensorFloatInternalScaling
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.nn import QuantConvTranspose1d, QuantConvTranspose2d

OUT_CH = 16
IN_CH = 8
IN_SHIFT = 5
IN_SCALE = 3
FEATURES = 5
KERNEL_SIZE = 3 
TOLERANCE = 1

QUANTIZERS = {
    'asymmetric_act_float': (Int8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'asymmetric_weight_float': (ShiftedUint8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'symmetric_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'symmetric_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint)}
BIAS_QUANTIZERS = {
    'bias_external_scale': (Int32Bias,),
    'bias_internal_scale': (Int8BiasPerTensorFloatInternalScaling,)}
QUANT_WBIOL_IMPL = [
    QuantLinear, QuantConv1d, QuantConv2d, QuantConvTranspose1d, QuantConvTranspose2d]
BIT_WIDTHS = [4, 8, 10] # below 8, equal 8, above 8
BIAS_BIT_WIDTHS = [8, 16, 32]


class TorchQuantWBIOLCases:

    @parametrize('impl', QUANT_WBIOL_IMPL, ids=[f'{c.__name__}' for c in QUANT_WBIOL_IMPL])
    @parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
    @parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
    @parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
    @parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
    @parametrize('bias_bit_width', BIAS_BIT_WIDTHS, ids=[f'b{b}' for b in BIAS_BIT_WIDTHS])
    @parametrize('bias_quantizer', BIAS_QUANTIZERS.values(), ids=list(BIAS_QUANTIZERS.keys()))
    @parametrize('per_channel', [True, False])
    def case_quant_wbiol_qcdq(
            self, impl, quantizers, input_bit_width, weight_bit_width, output_bit_width, bias_bit_width, bias_quantizer, per_channel, request):
        set_case_id(request.node.callspec.id, TorchQuantWBIOLCases.case_quant_wbiol_qcdq)  # Change the case_id based on current value of Parameters
        weight_quant, io_quant = quantizers
        bias_quantizer, = bias_quantizer  # pytest needs an iterable
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
                    bias_bit_width=bias_bit_width,
                    weight_scaling_per_output_channel=per_channel,
                    bias_quant=bias_quantizer,
                    return_quant_tensor=True)
                self.conv.weight.data.uniform_(-0.01, 0.01)

            def forward(self, x):
                return self.conv(x)

        torch.random.manual_seed(SEED)

        module = Model()
        return module
