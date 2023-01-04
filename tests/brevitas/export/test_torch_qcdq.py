import pytest

import torch
from pytest_cases import parametrize_with_cases, get_case_id
from tests.marker import requires_pt_ge
from brevitas.export import export_torch_qcdq

from .export_cases import TorchQuantWBIOLCases
from .export_cases import FEATURES, IN_CH, TOLERANCE, IN_MEAN, IN_SCALE


@parametrize_with_cases('model', cases=TorchQuantWBIOLCases.case_quant_wbiol_qcdq)
@requires_pt_ge('1.9.1')
def test_pytorch_qcdq_export(model, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    impl = case_id.split('-')[-1] # Inverse list of definition, 'impl' is -1.
    quantizer = case_id.split('-')[-2]
    input_bit_width = case_id.split('-')[-3]
    weight_bit_width = case_id.split('-')[-4]
    output_bit_width = case_id.split('-')[-5]
    bias_bit_width = case_id.split('-')[-6]
    bias_quantizer = case_id.split('-')[-7]
    
    if 'asymmetric_act' in quantizer and ('i10' in input_bit_width or 'o10' in output_bit_width):
        pytest.skip("Unsigned zero point supported on 8b or less.")
    if 'asymmetric_weight' in quantizer and 'w10' in weight_bit_width:
        pytest.skip("Unsigned zero point supported on 8b or less.")
    if 'internal_scale' in bias_quantizer and 'b32' in bias_bit_width:
        pytest.skip("This combination is prone to numerical errors as the scale gets too small.")

    if impl in ('QuantLinear'):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES)

    inp = torch.randn(in_size)
    model(inp) # Collect scale factors
    model.eval()
    export_path = f'torch_qcdq_{case_id}.pt'
    inp = torch.randn(in_size) * IN_SCALE + IN_MEAN  # redefine inp for testing
    out = model(inp)
    export_torch_qcdq(model, args=inp, export_path=export_path)
    pytorch_qcdq_model = torch.jit.load(export_path)
    torchscript_out = pytorch_qcdq_model(inp)
    torchscript_out_value = torchscript_out[0]
    tolerance = TOLERANCE * out.scale
    assert torch.allclose(out, torchscript_out_value, atol=tolerance)
