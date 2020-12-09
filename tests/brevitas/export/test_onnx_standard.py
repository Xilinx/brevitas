from operator import mul
from functools import reduce

import torch
import onnxruntime as rt
import numpy as np

from brevitas.nn import QuantConv2d, QuantReLU, QuantLinear, QuantIdentity, QuantMaxPool2d
from brevitas.inject.base import Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.inject.base import ShiftedUint8ActPerTensorFloat, ShiftedUint8WeightPerTensorFloat
from brevitas.onnx import export_standard_onnx

OUT_CH = 4
IN_CH = 5


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
    TOLERANCE = 2.1

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = QuantConv2d(
                out_channels=OUT_CH,
                in_channels=IN_CH,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=ShiftedUint8ActPerTensorFloat,
                output_quant=ShiftedUint8ActPerTensorFloat,
                return_quant_tensor=False)
            self.conv1.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv1(x)

    export_name = 'qlinearconv.onnx'
    inp = gen_linspaced_data(reduce(mul, IN_SIZE)).reshape(IN_SIZE)
    model = Model().eval()
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
    assert is_brevitas_ort_close(Model().eval(), inp, export_name)


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
    assert is_brevitas_ort_close(Model().eval(), inp, export_name)


def test_standard_onnx_quant_linear_export():
    IN_SIZE = (IN_CH, IN_CH)
    TOLERANCE = 1.1

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
    model = Model().eval()
    atol = model.linear.quant_output_scale().item() * TOLERANCE
    assert is_brevitas_ort_close(model, inp, export_name, atol=atol)