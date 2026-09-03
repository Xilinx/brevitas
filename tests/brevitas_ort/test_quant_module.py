# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul
import os

from hypothesis import given
from hypothesis import settings
from packaging.version import parse
import pytest
from pytest_cases import get_case_id
from pytest_cases import parametrize_with_cases
import torch

from brevitas import torch_version

from ..export_fixture import rm_onnx
from .common import *
from .quant_module_cases import build_wbiol_model
from .quant_module_cases import QuantAvgPoolCases
from .quant_module_cases import QuantRecurrentCases
from .quant_module_cases import wbiol_config_st
from .quant_module_cases import WBIOL_MAX_EXAMPLES


@settings(max_examples=WBIOL_MAX_EXAMPLES, deadline=None)
@given(config=wbiol_config_st())
def test_ort_wbiol(config):
    model = build_wbiol_model(config)
    rounding = config.rounding_type
    impl = config.impl.__name__
    quantizer = config.quantizer_name
    export_type = config.export_type
    onnx_opset = DEFAULT_ONNX_OPSET
    export_q_weight = False

    # Round weights can be exported as a Q-node (QuantizeLinear); floor weights and A2Q require
    # integer-initializer export instead, so they are excluded from Q-node export.
    if rounding == 'round' and 'a2q' not in quantizer:
        export_q_weight = True
    if 'fp8' in quantizer:
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
    export_name = f'qcdq_qop_export_{config.id}.onnx'
    try:
        close = is_brevitas_ort_close(
            model,
            inp,
            export_name,
            export_type,
            tolerance=INT_TOLERANCE,
            first_output_only=True,
            onnx_opset=onnx_opset,
            export_q_weight=export_q_weight)
    finally:
        rm_onnx(export_name)
    assert close


@parametrize_with_cases('model', cases=QuantAvgPoolCases)
@pytest.mark.parametrize('export_type', ['qcdq', 'qcdq_dynamo'])
def test_ort_avgpool(model, export_type, current_cases):
    if export_type == 'qcdq_dynamo' and torch_version < parse('2.8'):
        pytest.skip('QCDQ dynamo export requires PyTorch >= 2.8')
    in_size = (1, IN_CH, FEATURES, FEATURES)
    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_quant_avgpool_{export_type}.onnx'
    assert is_brevitas_ort_close(
        model, inp, export_name, export_type, tolerance=INT_TOLERANCE, first_output_only=True)
    rm_onnx(export_name)


@parametrize_with_cases('model', cases=QuantRecurrentCases)
@pytest.mark.parametrize('export_type', ['qcdq', 'qonnx_opset14'])
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
