# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnxruntime as ort
import pytest
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.infer_shapes import InferShapes
import torch

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_onnx_qop
from brevitas.export import export_qonnx
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantConvTranspose1d
from brevitas.nn import QuantConvTranspose2d
from brevitas.nn import QuantLinear
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.scaled_int import Int8AccumulatorAwareWeightQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat

SEED = 123456
OUT_CH = 16
IN_CH = 8
FEATURES = 5
INT_TOLERANCE = 2  # accept +2/-2 errors, required by per_channel/per_tensor, all 8 bit, quantconv1d/2d, qcdq, symmetric float
FLOAT_TOLERANCE = 1e-6
KERNEL_SIZE = 1  # keep float error during fake-quantization under control
BIT_WIDTHS = range(2, 9)
ACCUMULATOR_BIT_WIDTH_FOR_TESTS = 16


# For testing purpose, we create a custom quantizer with a reduced bitwidth for the accumulator
class A2QWeightQuantizerForTests(Int8AccumulatorAwareWeightQuant):
    accumulator_bit_width = ACCUMULATOR_BIT_WIDTH_FOR_TESTS


WBIOL_QUANTIZERS = {
    'asymmetric_per_tensor_float':
        (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_tensor_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'asymmetric_per_channel_float':
        (ShiftedUint8WeightPerChannelFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_channel_float': (Int8WeightPerChannelFloat, Int8ActPerTensorFloat),
    'a2q': (A2QWeightQuantizerForTests, Int8ActPerTensorFloat),
    'symmetric_per_tensor_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint),
    'symmetric_per_channel_fixed_point':
        (Int8WeightPerChannelFixedPoint, Int8ActPerTensorFixedPoint),
    'weight_symmetric_activation_dynamic_asymmetric_per_tensor_float':
        (Int8WeightPerTensorFloat, ShiftedUint8DynamicActPerTensorFloat)}
LSTM_QUANTIZERS = {
    'asymmetric_per_tensor_float':
        (ShiftedUint8WeightPerTensorFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_tensor_float': (Int8WeightPerTensorFloat, Int8ActPerTensorFloat),
    'asymmetric_per_channel_float':
        (ShiftedUint8WeightPerChannelFloat, ShiftedUint8ActPerTensorFloat),
    'symmetric_per_channel_float': (Int8WeightPerChannelFloat, Int8ActPerTensorFloat),
    'symmetric_per_tensor_fixed_point': (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint),
    'symmetric_per_channel_fixed_point':
        (Int8WeightPerChannelFixedPoint, Int8ActPerTensorFixedPoint)}
QUANT_WBIOL_IMPL = [
    QuantLinear, QuantConv1d, QuantConv2d, QuantConvTranspose1d, QuantConvTranspose2d]


def compute_ort(export_name, np_input):
    sess_opt = ort.SessionOptions()
    sess_opt.use_deterministic_compute = True  # Deterministic execution
    sess_opt.log_severity_level = 0  # Highest verbosity
    sess_opt.log_verbosity_level = 0  # Highest verbosity

    run_opt = ort.RunOptions()
    run_opt.log_severity_level = 0  # Highest verbosity
    run_opt.log_verbosity_level = 0  # Highest verbosity

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
        return all([
            recursive_allclose(ort_o, brevitas_o, tolerance) for ort_o,
            brevitas_o in zip(flat_ort_output, flat_brevitas_output)])
    ort_output = torch.from_numpy(ort_output)
    ort_output = ort_output.type_as(brevitas_output)
    if tolerance is not None:
        return torch.allclose(brevitas_output, ort_output, atol=tolerance)
    else:
        return (brevitas_output - ort_output == 0).all()


def is_brevitas_ort_close(
        model, np_input, export_name, export_type, tolerance=None, first_output_only=False):
    input_t = torch.from_numpy(np_input)
    with torch.no_grad():
        brevitas_output = model(input_t)

    if tolerance is not None and export_type == 'qcdq':
        tolerance = tolerance * brevitas_output.scale  # Float Output, tolerance is +/- output scale

    if export_type == 'qonnx':
        exported_model = export_qonnx(model, input_t, export_path=export_name)
        exported_model = ModelWrapper(exported_model)
        exported_model = exported_model.transform(InferShapes())
        idict = {exported_model.graph.input[0].name: np_input}
        odict = oxe.execute_onnx(exported_model, idict, True)
        ort_output = odict[exported_model.graph.output[0].name]
    else:
        if export_type == 'qop':
            export_onnx_qop(model, input_t, export_path=export_name)
            brevitas_output = brevitas_output.int(float_datatype=False)
        elif export_type == 'qcdq':
            export_onnx_qcdq(model, input_t, export_path=export_name)
        elif export_type == 'qcdq_opset14':
            export_onnx_qcdq(model, input_t, opset_version=14, export_path=export_name)
        elif export_type == 'qonnx_opset14':
            export_qonnx(model, input_t, opset_version=14, export_path=export_name)
        else:
            raise RuntimeError(f"Export type {export_type} not recognized.")

        ort_output = compute_ort(export_name, np_input)

    if first_output_only:
        if isinstance(ort_output, (tuple, list)):
            ort_output = ort_output[0]
        if isinstance(brevitas_output, tuple):
            brevitas_output = brevitas_output[0]
        # make sure we are not comparing 0s
        if (ort_output == 0).all() and (brevitas_output == 0).all():
            pytest.skip("Skip testing against all 0s.")

    return recursive_allclose(ort_output, brevitas_output, tolerance)


def gen_linspaced_data(num_samples, min_val=-1.0, max_val=1.0):
    return np.linspace(min_val, max_val, num_samples).astype(dtype=np.float32)
