import torch
import brevitas.nn as qnn
import torch.nn as nn
from brevitas.graph.calibrate import calibration_mode, bias_correction_mode
from brevitas.quant import Int8ActPerTensorFixedPoint
from tests.brevitas.hyp_helper import float_tensor_random_size_st
from hypothesis import given
import math

IN_CH = 8
OUT_CH = 16
BATCH = 1

def compute_quantile(x, q):
    k = int(math.floor(.01 * q * x.numel() + 0.5))
    result = x.abs().view(-1).kthvalue(k).values
    return result

def reference_implementation_scale_factors_po2(x, q=99.999, min_val=torch.tensor(1e-10), int_scale=128.):
    quant = compute_quantile(x,q)
    quant = torch.max(min_val, quant)
    quant_float_to_int = torch.ceil(torch.log2(quant)) # Float to Int Implementation for PowerOfTwo scale
    
    scale =  torch.pow(torch.tensor(2.), quant_float_to_int)/int_scale
    
    return scale


@given(inp=float_tensor_random_size_st())
def test_scale_factors_ptq_calibration_po2(inp):
    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)

        def forward(self, x):
            return self.act(x)
        
    model = TestModel()
    model.eval()
    with torch.no_grad():
        with calibration_mode(model):
            model(inp)

    expected_scale = reference_implementation_scale_factors_po2(inp)
    scale = model.act.quant_act_scale()

    assert torch.allclose(expected_scale, scale)


def test_calibration_training_state():
    
    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)

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


def test_bias_correction():
    inp = torch.randn(BATCH, IN_CH)
    fp_layer = nn.Linear(IN_CH, OUT_CH, bias=False)

    quant_layer = qnn.QuantLinear(IN_CH, OUT_CH, bias=False)
    quant_layer.weight.data = fp_layer.weight.data
    fp_layer.eval()
    quant_layer.eval()

    quant_out = quant_layer(inp)
    quant_fp = fp_layer(inp)

    with bias_correction_mode(quant_layer):
        quant_layer(inp)

    assert quant_layer.bias is not None
    assert torch.allclose(quant_layer.bias, quant_fp-quant_out)
