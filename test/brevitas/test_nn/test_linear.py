from torch.nn import Module
from brevitas.nn import QuantLinear
from dependencies import Injector


class Config(Injector):
    quant_type = 'INT'
    scaling_impl_type = 'STATS'
    scaling_stats_op = 'MAX'
    restrict_scaling_type = 'FP'
    bit_width_impl_type = 'CONST'
    per_channel_scaling = True
    narrow_range = True
    signed = True
    scaling_min_val = None
    bit_width = 8


OUTPUT_FEATURES = 10
INPUT_FEATURES = 5
BIT_WIDTH = 5

class Test(Module):

    def forward(self):
        return Test


class TestQuantLinear:

    def test_module(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            weight_quant=Config)
        print(mod.weight_quant.tensor_quant)

