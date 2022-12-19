from pytest_cases import parametrize, set_case_id
from torch import nn
import torch

from brevitas.quant.scaled_int import Int32Bias
from ...conftest import SEED


from brevitas.nn import QuantLinear, QuantConv1d, QuantConv2d
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.nn import QuantConvTranspose1d, QuantConvTranspose2d

OUT_CH = 16
IN_CH = 8
FEATURES = 5
KERNEL_SIZE = 3 
TOLERANCE = 1

QUANTIZERS = {
    'asymmetric_float': (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'symmetric_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint)}
QUANT_WBIOL_IMPL = [
    QuantLinear, QuantConv1d, QuantConv2d, QuantConvTranspose1d, QuantConvTranspose2d]

class TorchQuantWBIOLCases:

    @parametrize('impl', QUANT_WBIOL_IMPL, ids=[f'{c.__name__}' for c in QUANT_WBIOL_IMPL])
    @parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
    def case_quant_wbiol_qcdq(
            self, impl, quantizers, request):
        set_case_id(request.node.callspec.id, TorchQuantWBIOLCases.case_quant_wbiol_qcdq) # Change the case_id based on current value of Parameters
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
                    bias_quant=Int32Bias,
                    return_quant_tensor=True)
                self.conv.weight.data.uniform_(-0.01, 0.01)

            def forward(self, x):
                return self.conv(x)

        torch.random.manual_seed(SEED)

        module = Model()
        return module
