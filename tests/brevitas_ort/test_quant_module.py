# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul
import os

from packaging.version import parse
import pytest
from pytest_cases import get_case_id
from pytest_cases import parametrize_with_cases
import torch

from brevitas import torch_version
from tests.marker import requires_pt_ge

from ..export_fixture import rm_onnx
from .common import *
from .quant_module_cases import QuantAvgPoolCases
from .quant_module_cases import QuantRecurrentCases
from .quant_module_cases import QuantWBIOLCases


@parametrize_with_cases('model', cases=QuantWBIOLCases)
@pytest.mark.parametrize('export_type', ['qcdq', 'qonnx'])
@requires_pt_ge('1.10')
def test_ort_wbiol(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    rounding = case_id.split('-')[0]
    impl = case_id.split('-')[
        -3]  # Inverse list of definition, 'export_type' is -1, 'impl' is -2, etc.
    quantizer = case_id.split('-')[-7]
    o_bit_width = case_id.split('-')[-6]
    i_bit_width = case_id.split('-')[-4]
    onnx_opset = 14
    export_q_weight = False

    if rounding == 'round':
        export_q_weight = True

    if 'per_channel' in quantizer and 'asymmetric' in quantizer:
        pytest.skip('Per-channel zero-point is not well supported in ORT.')
    if 'QuantLinear' in impl and 'asymmetric' in quantizer:
        pytest.skip('ORT execution is unreliable and fails randomly on a subset of cases.')
    if 'dynamic' in quantizer and ((o_bit_width != "o8" or i_bit_width != "i8") or
                                   export_type != "qcdq"):
        pytest.skip('Dynamic Act Quant supported only for 8bit and QCDQ export')
    if export_type == 'qonnx' and 'fp8' in quantizer:
        pytest.skip('FP8 export requires QCDQ')
    if torch_version < parse('2.1') and 'fp8' in quantizer:
        pytest.skip('FP8 requires PyTorch 2.1 or higher')
    elif torch_version >= parse('2.1') and 'fp8' in quantizer:
        onnx_opset = 19
        export_q_weight = True

    if impl in ('QuantLinear'):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    elif impl in ('QuantConv2d', 'QuantConvTranspose2d'):
        in_size = (1, IN_CH, FEATURES, FEATURES)
    elif impl in ('QuantConv3d', 'QuantConvTranspose3d'):
        in_size = (1, IN_CH, FEATURES, FEATURES, FEATURES)
    else:
        raise RuntimeError(f"Unsupported operation {impl}")

    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)

    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_qop_export_{case_id}.onnx'
    assert is_brevitas_ort_close(
        model,
        inp,
        export_name,
        export_type,
        tolerance=INT_TOLERANCE,
        first_output_only=True,
        onnx_opset=onnx_opset,
        export_q_weight=export_q_weight)

    rm_onnx(export_name)


@parametrize_with_cases('model', cases=QuantAvgPoolCases)
@requires_pt_ge('1.10')
def test_ort_avgpool(model, current_cases):
    in_size = (1, IN_CH, FEATURES, FEATURES)
    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = 'qcdq_quant_avgpool.onnx'
    assert is_brevitas_ort_close(
        model, inp, export_name, 'qcdq', tolerance=INT_TOLERANCE, first_output_only=True)
    rm_onnx(export_name)


@parametrize_with_cases('model', cases=QuantRecurrentCases)
@pytest.mark.parametrize('export_type', ['qcdq', 'qonnx_opset14'])
@requires_pt_ge('1.10')
def test_ort_lstm(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    if 'a2q' in case_id:
        pytest.skip("A2Q doesn't support LSTM export currently.")

    if 'quant' in case_id and export_type == 'qonnx_opset14':
        pytest.skip(
            'Execution of quantized LSTM not supported out of the box for QONNX IR + ORT (requires qonnx lib).'
        )

    in_size = (FEATURES, 1, IN_CH)  # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    model.eval()
    export_name = f'lstm_export_{case_id}.onnx'
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=FLOAT_TOLERANCE)
    rm_onnx(export_name)
