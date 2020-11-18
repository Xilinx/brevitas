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

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops import min_int
from .utils import StatelessBuffer


class ZeroZeroPoint(brevitas.jit.ScriptModule):

    def __init__(self) -> None:
        super(ZeroZeroPoint, self).__init__()
        self.zero_point = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        return self.zero_point()


class MinUintZeroPoint(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, float_to_int_impl: Module, stats_reduce_dim: Optional[int]) -> None:
        super(MinUintZeroPoint, self).__init__()
        self.float_to_int_impl = float_to_int_impl
        self.stats_reduce_dim = stats_reduce_dim
        self.zero = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            min_val = torch.min(x)
        else:
            min_val = torch.min(x, dim=self.stats_reduce_dim)[0]
        min_val = torch.where(min_val <= self.zero(), min_val, self.zero())
        return - self.float_to_int_impl(min_val / scale)


class ShiftIntToUintZeroPoint(brevitas.jit.ScriptModule):

    @brevitas.jit.script_method
    def forward(self, x: Tensor, scale: Tensor, bit_width: Tensor) -> Tensor:
        return - min_int(signed=True, narrow_range=False, bit_width=bit_width)
