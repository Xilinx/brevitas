from operator import mul
from functools import reduce
from platform import system
from packaging.version import parse

import torch
import onnxruntime as rt
import numpy as np
import pytest

from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity, QuantMaxPool2d
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.onnx import export_standard_onnx
from brevitas import torch_version

ort_mac_fail = pytest.mark.skipif(
    parse(rt.__version__) == parse('1.6.0')
    and torch_version > parse('1.2.0')
    and system() == 'Darwin',
    reason='Issue with ORT 1.7.0 and QLinearMatMul on MacOS')

OUT_CH = 40
IN_CH = 50
TOLERANCE = 1.1


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)


def compute_ort(export_name, np_input):
    sess = rt.InferenceSession(export_name)
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_input})[0]
    return pred_onx


def is_brevitas_ort_close(model, np_input, export_name, atol=None):
    export_standard_onnx(model, input_shape=np_input.shape, export_path=export_name)
    brevitas_output = model(torch.from_numpy(np_input))
    ort_output = compute_ort(export_name, np_input)
    ort_output = torch.from_numpy(ort_output)
    assert (ort_output != 0.0).any()
    if atol is not None:
        return brevitas_output.isclose(ort_output, rtol=0.0, atol=atol).all()
    else:
        return brevitas_output.isclose(ort_output).all()


def test_standard_onnx_quant_conv():
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
                weight_quant=ShiftedUint8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-1.0, 1.0)

        def forward(self, x):
            return self.conv1(x)

    export_name = 'qlinearconv.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    atol = model.conv1.quant_output_scale().item() * TOLERANCE
    assert is_brevitas_ort_close(model, inp, export_name, atol=atol)


def test_standard_onnx_quant_identity_export():
    IN_SIZE = (1, OUT_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act = QuantIdentity(return_quant_tensor=False)

        def forward(self, x):
            return self.act(x)

    export_name = 'standard_identity.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    atol = model.act.quant_output_scale().item() * TOLERANCE
    assert is_brevitas_ort_close(model, inp, export_name, atol=atol)


def test_standard_onnx_quant_max_pool_export():
    IN_SIZE = (1, OUT_CH, IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.act = QuantIdentity(return_quant_tensor=True)
            self.pool = QuantMaxPool2d(kernel_size=2, return_quant_tensor=False)

        def forward(self, x):
            return self.pool(self.act(x))

    export_name = 'standard_maxpool.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    atol = model.act.quant_output_scale().item() * TOLERANCE
    assert is_brevitas_ort_close(model, inp, export_name, atol=atol)


@ort_mac_fail
def test_standard_onnx_quant_linear_export():
    IN_SIZE = (IN_CH, IN_CH)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = QuantLinear(
                in_features=IN_CH,
                out_features=OUT_CH,
                bias=False,
                weight_quant=ShiftedUint8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.linear.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.linear(x)

    export_name = 'standard_quant_linear.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model()
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    atol = model.linear.quant_output_scale().item() * TOLERANCE
    assert is_brevitas_ort_close(model, inp, export_name, atol=atol)
