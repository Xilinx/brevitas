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

from dataclasses import dataclass

import torch
from torch import Tensor

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_uint


@dataclass
class QuantTensor:
    value: Tensor
    scale: Tensor
    bit_width: Tensor
    signed: bool

    @staticmethod
    def check_input_type(other):
        if not isinstance(other, QuantTensor):
            raise RuntimeError("Other tensor is not a QuantTensor")

    def check_scaling_factors_same(self, other):
        if not torch.allclose(self.scale, other.scale):
            raise RuntimeError("Scaling factors are different")

    def view(self, *args, **kwargs):
        output = QuantTensor(
            self.value.view(*args, **kwargs), self.scale, self.bit_width, self.signed)
        return output

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        if self.signed:
            return QuantTensor(- self.value, self.scale, self.bit_width, self.signed)
        else:
            return QuantTensor(- self.value, self.scale, self.bit_width + 1, signed=True)

    def __add__(self, other):
        QuantTensor.check_input_type(other)
        self.check_scaling_factors_same(other)
        output_value = self.value + other.value
        output_scale = (self.scale + other.scale) / 2
        max_uint_val = max_uint(narrow_range=False, bit_width=self.bit_width)
        max_uint_val += max_uint(narrow_range=False, bit_width=other.bit_width)
        output_bit_width = ceil_ste(torch.log2(max_uint_val))
        output_signed = self.signed or other.signed
        output = QuantTensor(output_value, output_scale, output_bit_width, output_signed)
        return output

    def __mul__(self, other):
        QuantTensor.check_input_type(other)
        output_value = self.value * other.value
        output_scale = self.scale * other.scale
        output_bit_width = self.bit_width + other.bit_width
        output = QuantTensor(output_value, output_scale, output_bit_width)
        return output

    def __sub__(self, other):
        return self.__add__(- other)

    def __truediv__(self, other):
        QuantTensor.check_input_type(other)
        output_tensor = self.value / other.tensor
        output_scale = self.scale / other.scale
        output_bit_width = self.bit_width - other.bit_width
        output = QuantTensor(output_tensor, output_scale, output_bit_width)
        return output

    def __abs__(self):
        return QuantTensor(torch.abs(self.tensor), self.scale, self.bit_width, self.signed)

    def __pos__(self):
        return self