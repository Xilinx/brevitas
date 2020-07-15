from torch.nn import Module
from brevitas.nn import QuantConv2d
from dependencies import Injector

import torch

OUTPUT_CHANNELS = 10
INPUT_CHANNELS = 5
KERNEL_SIZE = (3, 3)
WEIGHT_BIT_WIDTH = 5


class TestQuantLinear:

    def test_module_init(self):
        mod = QuantConv2d(
            out_channels=OUTPUT_CHANNELS,
            in_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            bias=False)