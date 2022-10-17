import pytest

import torch
import onnxruntime as ort
import numpy as np

from brevitas.nn import QuantLinear, QuantConv1d, QuantConv2d
from brevitas.nn import QuantConvTranspose1d, QuantConvTranspose2d
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.export import export_standard_qop_onnx, export_standard_qcdq_onnx


OUT_CH = 16
IN_CH = 8
FEATURES = 5
TOLERANCE = 1  # accept +1/-1 errors
KERNEL_SIZE = 1  # keep float error during fake-quantization under control
BIT_WIDTHS = range(2, 9)
QUANTIZERS = {
 #   'asymmetric_float': (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'symmetric_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint)}
QUANT_WBIOL_IMPL = [
    QuantLinear, QuantConv1d, QuantConv2d, QuantConvTranspose1d, QuantConvTranspose2d]


def compute_ort(export_name, np_input):
    sess = ort.InferenceSession(export_name)
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_input})[0]
    return pred_onx


def is_brevitas_ort_close(model, np_input, export_name, export_type, tolerance=None):
    input_t = torch.from_numpy(np_input)
    if export_type == 'qop':
        export_standard_qop_onnx(model, input_t, export_path=export_name)
    elif export_type == 'qcdq':
        export_standard_qcdq_onnx(model, input_t, export_path=export_name)
    else:
        raise RuntimeError(f"Export type {export_type} not recognized.")
    brevitas_output = model(input_t).int(float_datatype=False)
    ort_output = compute_ort(export_name, np_input)
    ort_output = torch.from_numpy(ort_output)
    if (ort_output == 0).all() and (brevitas_output == 0).all(): # make sure we are not comparing 0s
        pytest.skip("Skip testing against all 0s.")
    if tolerance is not None:
        return (torch.abs(brevitas_output - ort_output) <= tolerance).all()
    else:
        return (brevitas_output - ort_output == 0).all()


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)