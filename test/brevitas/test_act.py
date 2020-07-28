import torch
from brevitas.nn import QuantReLU
from dependencies import Injector


class TestQuantReLU:

    def test_module_init_default(self):
        mod = QuantReLU(max_val=6)

    def test_module_init_const_scaling(self):
        mod = QuantReLU(max_val=6, scaling_impl_type='CONST')
