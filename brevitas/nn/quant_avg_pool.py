# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Callable, Union, Type
import math

import torch
from torch import Tensor
from torch.nn import AvgPool2d
from dependencies import Injector

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_uint
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.solver import update_trunc_quant_injector
from brevitas.inject.defaults import DefaultTruncQuantInjector as DefaultTruncQI
from .mixin.base import QuantLayerMixin
from .mixin.acc import QuantTruncMixin


class QuantAvgPool2d(QuantTruncMixin, QuantLayerMixin, AvgPool2d):

    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            trunc_quant: Union[AccQuantProxyProtocol, Type[Injector]] = DefaultTruncQI,
            return_quant_tensor: bool = True,
            update_injector: Callable = update_trunc_quant_injector,
            **kwargs):
        AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantTruncMixin.__init__(
            self,
            trunc_quant=trunc_quant,
            update_injector=update_injector,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    def forward(self, x: Union[Tensor, QuantTensor]):
        x = self.unpack_input(x)
        if self.export_mode:
            return self.export_handler(x.value)
        x = x.set(value=super(QuantAvgPool2d, self).forward(x.value))
        if self.is_trunc_quant_enabled:
            assert x.is_valid  # check input quant tensor is propertly formed
            rescaled_value = x.value * (self.kernel_size * self.kernel_size)  # remove avg scaling
            x = x.set(value=rescaled_value)
            x = x.set(bit_width=self.max_acc_bit_width(x.bit_width))
            x = self.trunc_quant(x)
        return self.pack_output(x)

    def max_acc_bit_width(self, input_bit_width):
        max_uint_input = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_uint_output = max_uint_input * self.kernel_size * self.kernel_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
