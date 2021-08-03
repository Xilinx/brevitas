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

from typing import Optional, Union, Type, Optional

from torch.nn import Module

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from brevitas.quant_tensor import QuantTensor
from .mixin.base import QuantLayerMixin
from .mixin.acc import QuantTruncMixin, QuantClampMixin, AccQuantType


class TruncQuantAccumulator(QuantTruncMixin, QuantLayerMixin, Module):

    def __init__(
            self,
            trunc_quant: Optional[AccQuantType] = None,
            return_quant_tensor: bool = True,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantTruncMixin.__init__(
            self,
            trunc_quant=trunc_quant,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    def forward(self, input: QuantTensor):
        x = self.unpack_input(input)
        x = self.trunc_quant(x)
        return self.pack_output(x)


class ClampQuantAccumulator(QuantClampMixin, QuantLayerMixin, Module):

    def __init__(
            self,
            clamp_quant: Optional[AccQuantType] = None,
            return_quant_tensor: bool = True,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantClampMixin.__init__(
            self,
            clamp_quant=clamp_quant,
            **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    def forward(self, input: QuantTensor):
        x = self.unpack_input(input)
        x = self.clamp_quant(x)
        return self.pack_output(x)