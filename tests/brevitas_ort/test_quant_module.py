from operator import mul
from functools import reduce

import torch
import pytest

from .common import *
from pytest_cases import parametrize_with_cases
from .quant_module_cases import QuantWBIOLCases, QuantRecurrentCases
from pytest_cases import parametrize_with_cases, get_case_id
from tests.marker import requires_pt_ge



@parametrize_with_cases('model', cases=QuantWBIOLCases)
@pytest.mark.parametrize('export_type', ['qcdq', 'qop'])
@requires_pt_ge('1.8.1')
def test_ort(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    impl = case_id.split('-')[-2] # Inverse list of definition, 'export_type' is -1, 'impl' is -2, etc.
    per_channel = case_id.split('-')[-6]
    quantizer = case_id.split('-')[-7]
    
    if impl in ('QuantConvTranspose1d', 'QuantConvTranspose2d') and export_type == 'qop':
        pytest.skip('Export of ConvTranspose is not supported for QOperation')
    if 'True' in per_channel and 'asymmetric' in quantizer:
        pytest.skip('Per-channel zero-point is not well supported in ORT.')
        
    if impl in ('QuantLinear'):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES)
    
    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)

    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name=f'qcdq_qop_export_{case_id}.onnx'
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=INT_TOLERANCE, first_output_only=True)
    
    
@parametrize_with_cases('model', cases=QuantRecurrentCases)
@pytest.mark.parametrize('export_type', ['qcdq_opset14', 'qonnx_opset14'])
@requires_pt_ge('1.10')
def test_ort_lstm(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    
    if 'quant' in case_id and export_type == 'qonnx_opset14':
        pytest.skip('Execution of quantized LSTM not supported out of the box for QONNX IR + ORT (requires qonnx lib).')    

    in_size = (FEATURES, 1, IN_CH) # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    model.eval()
    export_name = f'lstm_export_{case_id}.onnx'
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance=FLOAT_TOLERANCE)
