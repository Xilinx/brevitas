import pytest

import torch
import onnxruntime as ort
import numpy as np

from brevitas.nn import QuantLinear, QuantConv1d, QuantConv2d, QuantLSTM
from brevitas.nn import QuantConvTranspose1d, QuantConvTranspose2d
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.export import export_onnx_op, export_onnx_qcdq, export_qonnx


SEED = 123456
OUT_CH = 16
IN_CH = 8
FEATURES = 5
INT_TOLERANCE = 2  # accept +2/-2 errors, required by per_channel/per_tensor, all 8 bit, quantconv1d/2d, qcdq, symmetric float
FLOAT_TOLERANCE = 1e-6
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
    pred_onx = sess.run(None, {input_name: np_input}, run_options=run_opt)
    return pred_onx


def flatten(data):
    if isinstance(data, (tuple, list)):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def recursive_allclose(ort_output, brevitas_output, tolerance):
    if isinstance(ort_output, (list, tuple)) or isinstance(brevitas_output, (list, tuple)):
        flat_ort_output = tuple(flatten(ort_output))
        flat_brevitas_output = tuple(flatten(brevitas_output))
        return all([recursive_allclose(ort_o, brevitas_o, tolerance) for ort_o, brevitas_o in zip(flat_ort_output, flat_brevitas_output)])
    ort_output = torch.from_numpy(ort_output)
    ort_output = ort_output.type_as(brevitas_output)
    if tolerance is not None:
        return torch.allclose(brevitas_output, ort_output, atol=tolerance)
    else:
        return (brevitas_output - ort_output == 0).all()
    

def is_brevitas_ort_close(model, np_input, export_name, export_type, tolerance=None, first_output_only=False):
    input_t = torch.from_numpy(np_input)
    brevitas_output = model(input_t)
    
    if export_type == 'qop':
        export_onnx_op(model, input_t, export_path=export_name)
        brevitas_output = brevitas_output.int(float_datatype=False)
    elif export_type == 'qcdq':
        export_onnx_qcdq(model, input_t, export_path=export_name)
    elif export_type == 'qcdq_opset14':
        export_onnx_qcdq(model, input_t, opset_version=14, export_path=export_name)
    elif export_type == 'qonnx_opset14':
        export_qonnx(model, input_t, opset_version=14, export_path=export_name)
    else:
        raise RuntimeError(f"Export type {export_type} not recognized.")

    if tolerance is not None and export_type == 'qcdq':
        tolerance = tolerance * brevitas_output.scale # Float Output, tolerance is +/- output scale

    ort_output = compute_ort(export_name, np_input)
    
    if first_output_only:
        if isinstance(ort_output, tuple):
            ort_output = ort_output[0]
        if isinstance(brevitas_output, tuple):
            brevitas_output = brevitas_output[0]
    
    # make sure we are not comparing 0s
    if ort_output == 0 and (brevitas_output == 0).all(): 
        pytest.skip("Skip testing against all 0s.")

    return recursive_allclose(ort_output, brevitas_output, tolerance)


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)
