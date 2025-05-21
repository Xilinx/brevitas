
import torch

import brevitas.nn as qnn
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat

class Uint8DynamicActPerTensorFloat(Int8DynamicActPerTensorFloat):
    signed = False
    narrow_range = True

class Uint7DynamicActPerTensorFloat(Uint8DynamicActPerTensorFloat):
    bit_width=7

def test_scale_quant(model):
    uint8 = qnn.QuantIdentity(act_quant=Uint8DynamicActPerTensorFloat)
    uint7 = qnn.QuantIdentity(act_quant=Uint7DynamicActPerTensorFloat)
    layers_tested = 0
    layers_passed = 0
    layers_failed = 0
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantLinear):
            try:
                weight_scale = module.quant_weight().scale
                uint8.to(device=weight_scale.device)
                uint7.to(device=weight_scale.device)
                assert (weight_scale == uint8(weight_scale)).all()
                assert not (weight_scale == uint7(weight_scale)).all()
                layers_passed += 1
            except:
                layers_failed += 1
            layers_tested += 1
    print(f"Layers passed: {layers_passed}, Layers failed: {layers_failed}, Layers tested: {layers_tested}")
