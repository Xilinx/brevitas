
import torch

import brevitas.nn as qnn
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat

class Fp8e4m3OCPActPerTensorFloatConst(Fp8e4m3OCPActPerTensorFloat):
    scaling_impl_type = "const"
    scaling_init = 448.0

class Fp8e5m2OCPActPerTensorFloatConst(Fp8e4m3OCPActPerTensorFloatConst):
    exponent_bit_width = 5
    mantissa_bit_width = 2

def test_scale_quant(model):
    e4m3 = qnn.QuantIdentity(act_quant=Fp8e4m3OCPActPerTensorFloatConst)
    e5m2 = qnn.QuantIdentity(act_quant=Fp8e5m2OCPActPerTensorFloatConst)
    x = torch.rand((100,100))
    layers_tested = 0
    layers_passed = 0
    layers_failed = 0
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantLinear):
            try:
                weight_scale = module.quant_weight().scale
                e4m3.to(device=weight_scale.device)
                e5m2.to(device=weight_scale.device)
                x = x.to(device=weight_scale.device)
                assert (weight_scale == e4m3(weight_scale)).all()
                assert not (weight_scale == e5m2(weight_scale)).all()
                module.input_quant.return_quant_tensor = True
                act_scale = module.input_quant(x).scale
                assert (act_scale == e4m3(act_scale)).all()
                assert not (act_scale == e5m2(act_scale)).all()
                layers_passed += 1
            except:
                layers_failed += 1
            layers_tested += 1
    print(f"Layers passed: {layers_passed}, Layers failed: {layers_failed}, Layers tested: {layers_tested}")
