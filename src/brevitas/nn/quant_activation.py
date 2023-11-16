# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from torch import nn

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.inject.defaults import Uint8ActPerTensorFloat

from .quant_layer import ActQuantType
from .quant_layer import QuantNonLinearActLayer as QuantNLAL


# Starting from Torch 2.0, nn.Module init function accepts custom *args and **kwargs
# torch.nn.Sigmoid does not provide its own init method, and the presence of *args + **kwargs
# conflicts with the dependency injection package
class Sigmoid(nn.Sigmoid):

    def __init__(self):
        super().__init__()


# Starting from Torch 2.0, nn.Module init function accepts custom *args and **kwargs
# torch.nn.Tanh does not provide its own init method, and the presence of *args + **kwargs
# conflicts with the dependency injection package
class Tanh(nn.Tanh):

    def __init__(self):
        super().__init__()


class QuantReLU(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.ReLU,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantSigmoid(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=Sigmoid,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantTanh(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=Tanh,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantHardTanh(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloatMinMaxInit,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.Hardtanh,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantIdentity(QuantNLAL):

    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            input_quant=None,
            act_impl=None,
            passthrough_act=True,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
