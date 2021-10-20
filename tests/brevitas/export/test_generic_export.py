import torch

from brevitas.nn import QuantConv2d, QuantLinear, QuantAvgPool2d, QuantIdentity, QuantReLU, QuantMaxPool2d
from brevitas.quant.scaled_int import Int4WeightPerTensorFloatDecoupled
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int16Bias
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.export import export_brevitas_onnx, enable_debug
from brevitas_examples import imagenet_classification


OUT_CH = 50
IN_CH = 40
TOLERANCE = 1.1


def test_generic_quant_linear_export():
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp, export_path='generic_quant_linear.onnx')


def test_generic_decoupled_quant_linear_export():
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                weight_quant=Int4WeightPerTensorFloatDecoupled,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp, export_path='generic_decoupled_quant_linear.onnx')


def test_generic_quant_conv_export():
    IN_SIZE = (2, IN_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                bias=True,
                kernel_size=3,
                input_quant=Int8ActPerTensorFloat,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp, export_path='generic_quant_conv.onnx')


def test_generic_quant_tensor_export():
    IN_SIZE = (2, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.quant_inp = QuantIdentity(return_quant_tensor=True)
            self.linear = QuantLinear(
                out_features=OUT_CH,
                in_features=IN_CH,
                bias=True,
                output_quant=Int8ActPerTensorFloat,
                bias_quant=Int16Bias,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(self.quant_inp(x))

    inp = torch.randn(IN_SIZE)
    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp, export_path='generic_quant_tensor.onnx')


def test_generic_quant_avgpool_export():
    IN_SIZE = (2, OUT_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.inp_quant = QuantIdentity(return_quant_tensor=True)
            self.pool = QuantAvgPool2d(kernel_size=2, return_quant_tensor=False)

        def forward(self, x):
            return self.pool(self.inp_quant(x))

    inp = torch.randn(IN_SIZE)

    model = Model()
    model(inp)  # collect scale factors
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp, export_path='generic_quant_avgpool.onnx')


def test_generic_quant_avgpool_export_quant_input():
    IN_SIZE = (2, OUT_CH, IN_CH, IN_CH)
    inp = torch.randn(IN_SIZE)
    inp_quant = QuantIdentity(return_quant_tensor=True)
    model = QuantAvgPool2d(kernel_size=2, return_quant_tensor=False)
    inp_quant(inp)  # collect scale factors
    inp_quant.eval()
    model.eval()
    BrevitasONNXManager.export(
        model, input_t=inp_quant(inp), export_path='generic_quant_avgpool_quant_input.onnx')


def test_debug_brevitas_onnx_export():
    model, cfg = imagenet_classification.model_with_cfg('quant_mobilenet_v1_4b', pretrained=False)
    model.eval()
    debug_hook = enable_debug(model, proxy_level=True)
    input_tensor = torch.randn(1, 3, 224, 224)
    export_brevitas_onnx(model, input_t=input_tensor, export_path='generic_debug.onnx')
    model(input_tensor)
    assert debug_hook.values
