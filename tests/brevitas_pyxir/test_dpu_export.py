from operator import mul
from functools import reduce

import pytest
import torch
import numpy as np

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d, QuantEltwiseAdd
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPoint
from brevitas.export import DPUv1Manager, DPUv2Manager

from tests.marker import requires_pt_ge


OUT_CH = 40
IN_CH = 50
TOLERANCE = 1.1
DPUS = ['DPUv1', 'DPUv2']
MANAGERS_MAP = {'DPUv1': DPUv1Manager, 'DPUv2': DPUv2Manager}


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv1(x)

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_linear(dpu):
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_linear.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_bias(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=True,
                weight_quant=Int8WeightPerTensorFixedPoint,
                bias_quant=Int8BiasPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv1(x)

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_bias.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_standard_onnx_quant_linear_bias_export(dpu):
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=True,
                weight_quant=Int8WeightPerTensorFixedPoint,
                bias_quant=Int8BiasPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_linear_bias.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_max_pool(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=True)
            self.max_pool = QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.max_pool(self.conv1(x))

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_maxpool.onnx')


@requires_pt_ge('1.5.0')
@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_f_max_pool(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=True)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return torch.nn.functional.max_pool2d(self.conv1(x), kernel_size=2, stride=2)

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_f_maxpool.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_relu(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=True)
            self.relu = QuantReLU(act_quant=None, return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.relu(self.conv1(x))

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_relu.onnx')


@requires_pt_ge('1.5.0')
@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_f_relu(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=True)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return torch.nn.functional.relu(self.conv1(x))

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_f_relu.onnx')


@pytest.mark.parametrize('dpu', DPUS)
def test_dpu_export_onnx_quant_conv_add(dpu):
    FEATURES = 7
    IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    KERNEL_SIZE = 3

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=Int8ActPerTensorFixedPoint,
                return_quant_tensor=True)
            self.conv2 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                input_quant=Int8ActPerTensorFixedPoint,
                output_quant=self.conv1.output_quant,
                return_quant_tensor=True)
            self.add = QuantEltwiseAdd(input_quant=None, output_quant=Int8ActPerTensorFixedPoint)
            self.conv1.weight.data.uniform_(-0.01, 0.01)
            self.conv2.weight.data.uniform_(-0.0001, 0.0001)

        def forward(self, x):
            return self.add(self.conv1(x), self.conv2(x))

    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    MANAGERS_MAP[dpu].export(model, input_shape=IN_SIZE, export_path=f'{dpu}_conv_add.onnx')