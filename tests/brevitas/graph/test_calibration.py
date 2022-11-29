import torch
import brevitas.nn as qnn
import torch.nn as nn
from brevitas.graph.calibrate import calibration_mode,finalize_collect_stats
from brevitas.quant import Int8ActPerTensorFixedPoint
from tests.brevitas.hyp_helper import float_tensor_random_size_st
from hypothesis import given
import math

def compute_quantile(x, q):
    k = int(math.floor(.01 * q * x.numel() + 0.5))
    result = x.abs().view(-1).kthvalue(k).values
    return result

def reference_implementation_scale_factors_po2(x, q=99.999, min_val=torch.tensor(1e-10)):
    quant = compute_quantile(x,q)
    quant = torch.max(min_val, quant)
    quant_float_to_int = torch.ceil(torch.log2(quant)) # Float to Int Implementation for PowerOfTwo scale
    
    int_scale = 128. # Signed, PowerOfTwo int scale
    scale =  torch.pow(torch.tensor(2.), quant_float_to_int)/int_scale
    
    return scale

@given(inp=float_tensor_random_size_st())
def test_scale_factors_ptq_calibration_po2(inp):
    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(Int8ActPerTensorFixedPoint)

        def forward(self, x):
            return self.act(x)
        
    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            model(inp)
    model.apply(finalize_collect_stats)

    expected_scale = reference_implementation_scale_factors_po2(inp)
    scale = model.act.quant_act_scale()

    assert torch.allclose(expected_scale, scale)


def test_calibration_training_state():
    
    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(Int8ActPerTensorFixedPoint)

        def forward(self, x):
            return self.act(x)
        
    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            assert model.act.act_quant.training == True
            assert model.training == False
    
    assert model.act.act_quant.training == False
    assert model.training == False