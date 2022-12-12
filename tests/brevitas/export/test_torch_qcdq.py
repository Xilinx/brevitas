
import torch

from .export_cases import TorchQuantWBIOLCases
from .export_cases import FEATURES, IN_CH
from pytest_cases import parametrize_with_cases, get_case_id
from tests.marker import requires_pt_ge
from brevitas.export import export_torch_qcdq
import brevitas.nn as qnn

@parametrize_with_cases('model', cases=TorchQuantWBIOLCases.case_quant_wbiol_qcdq)
@requires_pt_ge('1.9.1')
def test_pytorch_qcdq_export(model, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    impl = case_id.split('-')[-1] # Inverse list of definition, 'impl' is -1.

    if impl in ('QuantLinear'):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES)

    inp = torch.randn(in_size)
    model(inp) # Collect scale factors
    model.eval()
    
    out = model(inp)
    export_torch_qcdq(model, args=inp, export_path='pytorch_qcdq.ts')
    pytorch_qcdq_model = torch.load('pytorch_qcdq.ts')
    torchscript_out = pytorch_qcdq_model(inp)
    
    assert torch.allclose(out, torchscript_out)
