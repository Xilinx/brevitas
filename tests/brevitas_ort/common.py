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
TOLERANCE = 2  # accept +2/-2 errors, required by per_channel/per_tensor, all 8 bit, quantconv1d/2d, qcdq, symmetric float
KERNEL_SIZE = 1  # keep float error during fake-quantization under control
BIT_WIDTHS = range(2, 9)
QUANTIZERS = {
    'asymmetric_float': (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'symmetric_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint)}
QUANT_WBIOL_IMPL = [
    QuantLinear, QuantConv1d, QuantConv2d, QuantConvTranspose1d, QuantConvTranspose2d]


def compute_ort(export_name, np_input):
    sess_opt = ort.SessionOptions()
    sess_opt.use_deterministic_compute=True # Deterministic execution
    sess_opt.log_severity_level = 0 # Highest verbosity
    sess_opt.log_verbosity_level = 0 # Highest verbosity

    run_opt = ort.RunOptions()
    run_opt.log_severity_level = 0 # Highest verbosity
    run_opt.log_verbosity_level = 0 # Highest verbosity

    sess = ort.InferenceSession(export_name, sess_opt)
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_input}, run_options=run_opt)[0]
    return pred_onx


def is_brevitas_ort_close(model, np_input, export_name, export_type, tolerance=None):
    input_t = torch.from_numpy(np_input)
    brevitas_output = model(input_t)
    if export_type == 'qop':
        export_standard_qop_onnx(model, input_t, export_path=export_name)
        brevitas_output = brevitas_output.int(float_datatype=False)
    elif export_type == 'qcdq':
        export_standard_qcdq_onnx(model, input_t, export_path=export_name)
    else:
        raise RuntimeError(f"Export type {export_type} not recognized.")

    if tolerance is not None and export_type == 'qcdq':
        tolerance = tolerance * brevitas_output.scale # Float Output, tolerance is +/- output scale

    ort_output = compute_ort(export_name, np_input)
    ort_output = torch.from_numpy(ort_output).type_as(brevitas_output)

    if (ort_output == 0).all() and (brevitas_output.value == 0).all(): # make sure we are not comparing 0s
        pytest.skip("Skip testing against all 0s.")
    if tolerance is not None:
        return torch.allclose(brevitas_output, ort_output, atol=tolerance)
    else:
        return (brevitas_output - ort_output == 0).all()


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)
