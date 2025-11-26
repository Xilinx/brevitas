
import torch
import brevitas
from brevitas import nn as qnn
from brevitas.quant.scaled_int import (
    Int8WeightPerTensorFloat, 
    Int8WeightPerTensorFloatMSE, 
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloatMSESparse
)
from brevitas.core.function_wrapper import RoundSte, SparseRoundSte

class MyRoundSte(RoundSte):
    def __init__(self, sparsity_ratio):
        self.sparsity_ratio=sparsity_ratio
        self.sparsity_mask=None
        super().__init__()

    @brevitas.jit.script_method
    def forward(self, x):
        print("Sparsity ratio: ", getattr(self, 'sparsity_ratio', None))
        if self.training:
            print("MyRoundSte in training mode")
        else:
            print("MyRoundSte in evaluation mode")
        print("Shape of x in MyRoundSte:", x.shape)
        return super().forward(x)

print(type(Int8WeightPerTensorFloat), Int8WeightPerTensorFloat)
qlinear = qnn.QuantLinear(5, 10, weight_quant=Int8WeightPerTensorFloatMSESparse,
                        input_quant=Int8ActPerTensorFloat,
                        return_quant_tensor=True,
                        weight_param_method="mse",
                        weight_quant_format='int',
                        weight_quant_granularity='per_channel',
                        weight_scale_precision='float_scale',
                        weight_sparsity_ratio="0.75")
print(qlinear)
x = torch.randn(1, 2, 5)
output = qlinear(x)
qlinear.eval()
output = qlinear(x)
