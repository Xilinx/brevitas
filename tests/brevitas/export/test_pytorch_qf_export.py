import torch

from brevitas.nn import QuantConv2d, QuantLinear
from brevitas.inject.defaults import Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.export.pytorch.manager import PytorchQuantManager

OUT_CH = 3
IN_CH = 2


def test_pytorch_quant_conv_export():
    IN_SIZE = (1, IN_CH, 7, 7)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Uint8ActPerTensorFloat,
                output_quant=Uint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.conv1.weight.data.uniform_(0.0, 0.01)

        def forward(self, x):
            return self.conv1(x)

    model = Model().eval()
    inp = torch.randn(IN_SIZE)
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model)
    pytorch_out = pytorch_qf_model(inp)
    pytorch_out = pytorch_out.dequantize().cpu()
    atol = brevitas_out.scale.item() * 2.1  # slightly larger than 2 scale factor
    brevitas_out = brevitas_out.value
    print(brevitas_out - pytorch_out)
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
                input_quant=Uint8ActPerTensorFloat,
                output_quant=Uint8ActPerTensorFloat,
                return_quant_tensor=True)
            self.linear.weight.data.uniform_(0.0, 0.01)

        def forward(self, x):
            return self.linear(x)

    model = Model().eval()
    inp = torch.randn(IN_SIZE)
    brevitas_out = model(inp)
    pytorch_qf_model = PytorchQuantManager.export(model)
    pytorch_out = pytorch_qf_model(inp)
    atol = brevitas_out.scale.item() * 1.1  # slightly larger than 1 scale factor
    brevitas_out = brevitas_out.value
    assert pytorch_out.dequantize().cpu().isclose(brevitas_out, rtol=0.0, atol=atol).all()