# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul
import os

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from packaging.version import parse
import pytest
from pytest_cases import get_case_id
from pytest_cases import parametrize_with_cases
import torch

from brevitas import torch_version
from brevitas.nn import QuantLSTM

from ..export_fixture import rm_onnx
from .common import *
from .quant_module_cases import build_wbiol_model
from .quant_module_cases import QuantAvgPoolCases
from .quant_module_cases import QuantRecurrentCases
from .quant_module_cases import WBIOL_BITWIDTH_EXAMPLES
from .quant_module_cases import wbiol_bitwidths_st
from .quant_module_cases import WBIOL_FLAG_COMBOS


# WBIOL hybrid: enumerate valid flag combinations (one xdist-distributed node each) and let
# Hypothesis sample the bit-widths within each node.
@pytest.mark.parametrize('flags', WBIOL_FLAG_COMBOS, ids=[f.id for f in WBIOL_FLAG_COMBOS])
@settings(max_examples=WBIOL_BITWIDTH_EXAMPLES, deadline=None)
@given(data=st.data())
def test_ort_wbiol(flags, data):
    bit_widths = data.draw(wbiol_bitwidths_st(flags))
    model = build_wbiol_model(flags, bit_widths)
    impl = flags.impl.__name__
    quantizer = flags.quantizer_name
    export_type = flags.export_type
    onnx_opset = 19 if 'fp8' in quantizer else DEFAULT_ONNX_OPSET
    export_q_weight = flags.export_q_weight

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
    export_name = f'qcdq_qop_export_{flags.id}.onnx'
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
@pytest.mark.parametrize(
    'export_type', ['qcdq', 'qcdq_dynamo', 'qonnx_opset14', 'qonnx_dynamo_opset18'])
def test_ort_lstm(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    if 'a2q' in case_id:
        pytest.skip("A2Q doesn't support LSTM export currently.")

    if 'dynamo' in export_type and torch_version < parse('2.8'):
        pytest.skip('Dynamo export requires PyTorch >= 2.8')

    if 'quant' in case_id and 'qonnx' in export_type:
        pytest.skip(
            'Execution of quantized LSTM not supported out of the box for QONNX IR + ORT (requires qonnx lib).'
        )

    # A batch size other than 1 catches a squeeze of the wrong dimension of the LSTM outputs
    in_size = (FEATURES, 2, IN_CH)  # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    model.eval()
    export_name = f'lstm_export_{case_id}.onnx'
    # Dynamo QCDQ only supports Q-node weight export
    export_q_weight = export_type == 'qcdq_dynamo'
    assert is_brevitas_ort_close(
        model,
        inp,
        export_name,
        export_type,
        tolerance=FLOAT_TOLERANCE,
        export_q_weight=export_q_weight)
    rm_onnx(export_name)


@pytest.mark.parametrize('export_type', ['qonnx', 'qonnx_dynamo'])
@pytest.mark.parametrize(
    'bidirectional, cifg, num_layers',
    [
        pytest.param(False, False, 1, id='unidirectional'),
        pytest.param(True, True, 2, id='bidirectional_cifg_2_layers'),
        pytest.param('shared_input_hidden', False, 2, id='shared_input_hidden_2_layers')])
@pytest.mark.parametrize(
    'quantizer', ['symmetric_per_tensor_float', 'asymmetric_per_channel_float'])
def test_ort_lstm_qonnx_weight_quant(
        request, quantizer, bidirectional, cifg, num_layers, export_type):
    # QONNX exports the weights through Quant nodes, so the graph runs on the qonnx executor
    if export_type == 'qonnx_dynamo' and torch_version < parse('2.8'):
        pytest.skip('Dynamo export requires PyTorch >= 2.8')

    torch.random.manual_seed(SEED)
    model = QuantLSTM(
        input_size=IN_CH,
        hidden_size=OUT_CH,
        weight_quant=LSTM_QUANTIZERS[quantizer][0],
        weight_bit_width=4,
        bias_quant=None,
        io_quant=None,
        gate_acc_quant=None,
        sigmoid_quant=None,
        tanh_quant=None,
        cell_state_quant=None,
        num_layers=num_layers,
        bidirectional=bool(bidirectional),
        shared_input_hidden_weights=bidirectional == 'shared_input_hidden',
        coupled_input_forget_gates=cifg)
    model.eval()
    in_size = (FEATURES, 2, IN_CH)  # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    export_name = f'lstm_qonnx_weight_quant_{request.node.callspec.id}.onnx'
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=FLOAT_TOLERANCE)
    rm_onnx(export_name)
