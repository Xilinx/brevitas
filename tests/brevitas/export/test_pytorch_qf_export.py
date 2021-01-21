import torch
import pytest

from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity, QuantReLU
from brevitas.inject.defaults import ShiftedUint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.export.pytorch.manager import PytorchQuantManager

OUT_CH = 50
IN_CH = 40
TOLERANCE = 1.1


# see https://github.com/pytorch/pytorch/projects/17#card-53197198
@pytest.mark.xfail(strict=True)
def test_pytorch_quant_linear_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model, inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.linear.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


def test_quant_act_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act1 = QuantIdentity(
                act_quant=ShiftedUint8ActPerTensorFloat, return_quant_tensor=True)
            self.act2 = QuantReLU(act_quant=Uint8ActPerTensorFloat)

        def forward(self, x):
            return self.act2(self.act1(x))

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model, inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.act2.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()