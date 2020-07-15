from torch.nn import Module
from brevitas.nn import QuantLinear
from dependencies import Injector

import torch

OUTPUT_FEATURES = 10
INPUT_FEATURES = 5
BIT_WIDTH = 5


class TestQuantLinear:

    def test_module_init(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=False)

    def test_forward(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True)
        x = torch.rand(size=(3, INPUT_FEATURES))
        mod(x)

    def test_override(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True, weight_scaling_impl_type='HE')
        print(mod.quant_weight_scale())
