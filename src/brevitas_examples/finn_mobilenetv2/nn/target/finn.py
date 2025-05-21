
from typing import Optional

import torch
from torch import nn

from brevitas.inject.defaults import Uint8ActPerTensorFloat

from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

class ReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.relu6(x)

class QuantReLU6(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=ReLU6,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
