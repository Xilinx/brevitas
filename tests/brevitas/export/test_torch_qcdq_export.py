
import torch
from operator import mul
from functools import reduce
# from brevitas.nn import QuantConv2d, QuantLinear, QuantAvgPool2d, QuantIdentity, QuantReLU, QuantMaxPool2d
# from brevitas.quant.scaled_int import Int4WeightPerTensorFloatDecoupled
# from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int16Bias
# from brevitas.export import export_standard_qcdq_onnx
# from brevitas.export import export_brevitas_onnx, enable_debug
# from brevitas_examples import imagenet_classification

from .export_cases import TorchQuantWBIOLCases
from .export_cases import FEATURES, IN_CH
from pytest_cases import parametrize_with_cases, get_case_id
from tests.marker import requires_pt_ge
from brevitas.export import export_torch_qcdq


@parametrize_with_cases('model', cases=TorchQuantWBIOLCases.case_quant_wbiol_qcdq)
@requires_pt_ge('1.8.1')
def test_ort(model, current_cases):
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
    
    model.eval()
    pytorch_qf_model = export_torch_qcdq(model, args=inp, export_path='pytorch_qcdq.pth')

