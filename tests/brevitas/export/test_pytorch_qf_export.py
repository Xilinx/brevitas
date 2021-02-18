import torch
import pytest

from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity, QuantReLU, QuantMaxPool2d
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.export.pytorch.manager import PytorchQuantManager

OUT_CH = 50
IN_CH = 40
TOLERANCE = 1.1


def test_pytorch_quant_conv_export():
    IN_SIZE = (2, IN_CH, IN_CH, IN_CH)
    KERNEL_SIZE = (3, 3)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                input_bit_width=7,
                output_bit_width=7,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model, inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.conv.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()


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
                input_bit_width=7,
                output_bit_width=7,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
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
                bit_width=7,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=True)
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


def test_quant_max_pool2d_export():
    IN_SIZE = (1, 1, IN_CH, IN_CH)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act = QuantIdentity(
                bit_width=7,
                act_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.pool = QuantMaxPool2d(
                kernel_size=KERNEL_SIZE, stride=KERNEL_SIZE,
                return_quant_tensor=False)

        def forward(self, x):
            return self.pool(self.act(x))

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model, inp)
    pytorch_out = pytorch_qf_model(inp)
    atol = model.act.quant_output_scale().item() * TOLERANCE
    assert pytorch_out.isclose(brevitas_out, rtol=0.0, atol=atol).all()