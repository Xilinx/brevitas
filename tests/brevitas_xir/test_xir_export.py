from operator import mul
from functools import reduce

import pytest
import torch
import numpy as np

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d, QuantEltwiseAdd
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPointInternalScaling
from brevitas.export import XIRManager

from tests.marker import requires_pt_ge


OUT_CH = 40
IN_CH = 50
TOLERANCE = 1.1


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)


def test_dpu_export_onnx_quant_conv():
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
    XIRManager.export(model, input_shape=IN_SIZE, export_path='xir_conv.xmodel')


def test_dpu_export_onnx_quant_linear():
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
    XIRManager.export(model, input_shape=IN_SIZE, export_path='xir_linear.xmodel')


def test_dpu_export_onnx_quant_conv_bias():
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
                bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
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
    XIRManager.export(model, input_shape=IN_SIZE, export_path='xir_conv_bias.xmodel')


def test_standard_onnx_quant_linear_bias_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=True,
                weight_quant=Int8WeightPerTensorFixedPoint,
                bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
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
    XIRManager.export(model, input_shape=IN_SIZE, export_path='xir_linear_bias.xmodel')