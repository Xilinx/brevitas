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
@pytest.mark.parametrize('export_type', ['qcdq', 'qcdq_dynamo', 'qonnx', 'qonnx_dynamo'])
@requires_pt_ge('1.10')
def test_ort_wbiol(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    # case_id has the form
    # 'quant_wbiol-{quantizer}-o#-w#-i#-{impl}-rtype_{round|floor}-{export_type}', so the
    # fields are indexed from the end: export_type=-1, rounding=-2, impl=-3, i_bit=-4,
    # w_bit=-5, o_bit=-6, quantizer=-7.
    rounding = case_id.split('-')[-2].replace('rtype_', '')
    impl = case_id.split('-')[-3]
    quantizer = case_id.split('-')[-7]
    o_bit_width = case_id.split('-')[-6]
    i_bit_width = case_id.split('-')[-4]
    onnx_opset = 14
    export_q_weight = False

    # Round weights can be exported as a Q-node (QuantizeLinear); floor weights and A2Q require
    # integer-initializer export instead, so they are excluded from Q-node export.
    if rounding == 'round' and 'a2q' not in quantizer:
        export_q_weight = True

    if export_type == 'qcdq_dynamo':
        # The dynamo (torch.export) QCDQ path only supports configs that don't rely on
        # data_ptr()-keyed integer export (i.e. quantized bias / integer-initializer
        # weights). In the WBIOL cases bias is quantized for everything except the fp8
        # and dynamic-activation quantizers, so we limit dynamo coverage to those.
        if torch_version < parse('2.8'):
            pytest.skip('QCDQ dynamo export requires PyTorch >= 2.8')
        if rounding != 'round':
            pytest.skip(
                'Dynamo QCDQ exports weights as a Q-node; QuantizeLinear supports only '
                'round-to-nearest-even, so non-round weight rounding is unsupported.')
        if 'fp8' not in quantizer and 'dynamic' not in quantizer:
            pytest.skip('QCDQ dynamo export does not support quantized bias (data_ptr export).')

    if export_type == 'qonnx_dynamo' and torch_version < parse('2.8'):
        pytest.skip('QONNX dynamo export requires PyTorch >= 2.8')

    if 'per_channel' in quantizer and 'asymmetric' in quantizer:
        pytest.skip('Per-channel zero-point is not well supported in ORT.')
    if 'QuantLinear' in impl and 'asymmetric' in quantizer:
        pytest.skip('ORT execution is unreliable and fails randomly on a subset of cases.')
    if 'dynamic' in quantizer and ((o_bit_width != "o8" or i_bit_width != "i8") or
                                   export_type not in ("qcdq", "qcdq_dynamo")):
        pytest.skip('Dynamic Act Quant supported only for 8bit and QCDQ export')
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
@pytest.mark.parametrize('export_type', ['qcdq', 'qcdq_dynamo'])
@requires_pt_ge('1.10')
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
