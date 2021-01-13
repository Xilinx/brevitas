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

from abc import ABC
from typing import Optional, NamedTuple

import torch
from torch import Tensor

from brevitas.function.ops_ste import ceil_ste, round_ste
from brevitas.function.ops import max_int


class QuantTensor(NamedTuple):
    value: Tensor
    scale: Optional[Tensor] = None
    zero_point: Optional[Tensor] = None
    bit_width: Optional[Tensor] = None
    signed: Optional[bool] = None
    training: Optional[bool] = None

    @property
    def tensor(self):
        return self.value

    @property
    def is_valid(self):
        return self.value is not None \
               and self.scale is not None \
               and self.zero_point is not None \
               and self.bit_width is not None \
               and self.signed is not None

    def set(self, **kwargs):
        return self._replace(**kwargs)

    def detach_(self):
        self.value.detach_()
        self.scale.detach_()
        self.zero_point.detach_()
        self.bit_width.detach_()

    def detach(self):
        return QuantTensor(
            self.value.detach() if self.value is not None else None,
            self.scale.detach() if self.scale is not None else None,
            self.zero_point.detach() if self.zero_point is not None else None,
            self.bit_width.detach() if self.bit_width is not None else None,
            self.signed)

    def int(self, float_datatype=False):
        if self.is_valid:
            int_value = self.value / self.scale
            int_value = int_value + self.zero_point
            int_value = round_ste(int_value)
            if float_datatype:
                return int_value
            else:
                return int_value.int()
        else:
            raise RuntimeError(f"QuantTensor not well formed, all fields must be set: {self}")

    @staticmethod
    def check_input_type(other):
        if not isinstance(other, QuantTensor):
            raise RuntimeError("Other tensor is not a QuantTensor")

    def check_scaling_factors_same(self, other):
        if self.training is not None and self.training:
            return True
        if not torch.allclose(self.scale, other.scale):
            raise RuntimeError("Scaling factors are different")

    def check_zero_points_same(self, other):
        if self.training is not None and self.training:
            return True
        if not torch.allclose(self.zero_point, other.zero_point):
            raise RuntimeError("Zero points are different")

    def check_bit_width_same(self, other):
        if not torch.allclose(self.bit_width, other.bit_width):
            raise RuntimeError("Bit widths are different")

    def check_sign_same(self, other):
        if not self.signed == other.signed:
            raise RuntimeError("Signs are different")

    def view(self, *args, **kwargs):
        return self.set(value= self.value.view(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.set(value=self.value.reshape(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return self.set(value=self.value.flatten(*args, **kwargs))

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)

    @property
    def shape(self):
        return self.value.shape

    def add(self, other):
        return self + other

    @staticmethod
    def cat(tensor_list, dim):
        assert len(tensor_list) >= 2, 'Two or more tensors required for concatenation'
        first_qt = tensor_list[0]
        if all([qt.is_valid for qt in tensor_list]):
            for qt in tensor_list[1:]:
                QuantTensor.check_input_type(qt)
                first_qt.check_scaling_factors_same(qt)
                first_qt.check_scaling_factors_same(qt)
                first_qt.check_bit_width_same(qt)
                first_qt.check_sign_same(qt)
            output_value = torch.cat([qt.value for qt in tensor_list], dim=dim)
            output_scale = sum([qt.scale for qt in tensor_list]) / len(tensor_list)
            output_zero_point = sum([qt.zero_point for qt in tensor_list]) / len(tensor_list)
            output_bit_width = sum([qt.bit_width for qt in tensor_list]) / len(tensor_list)
            output_signed = first_qt.signed # they are the same
            return QuantTensor(
                output_value, output_scale, output_zero_point, output_bit_width, output_signed)
        else:
            output_value = torch.cat([qt.value for qt in tensor_list], dim=dim)
            return QuantTensor(output_value)


    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        if self.signed:
            return QuantTensor(
                - self.value, self.scale, self.zero_point, self.bit_width, self.signed)
        else:
            return QuantTensor(
                - self.value, self.scale, self.bit_width + 1, signed=True)

    def __add__(self, other):
        QuantTensor.check_input_type(other)
        if self.is_valid and other.is_valid:
            self.check_scaling_factors_same(other)
            self.check_zero_points_same(other)
            output_value = self.value + other.value
            output_scale = (self.scale + other.scale) / 2
            output_zero_point = (self.zero_point + other.zero_point) / 2
            max_uint_val = max_int(signed=False, narrow_range=False, bit_width=self.bit_width)
            max_uint_val += max_int(signed=False, narrow_range=False, bit_width=other.bit_width)
            output_bit_width = ceil_ste(torch.log2(max_uint_val))
            output_signed = self.signed or other.signed
            output = QuantTensor(
                output_value, output_scale, output_zero_point, output_bit_width, output_signed)
        else:
            output_value = self.value + other.value
            output = QuantTensor(output_value)
        return output

    def __mul__(self, other):  # todo zero point
        QuantTensor.check_input_type(other)
        if self.is_valid and other.is_valid:
            output_value = self.value * other.value
            output_scale = self.scale * other.scale
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output = QuantTensor(output_value, output_scale, output_bit_width, output_signed)
        else:
            output_value = self.value * other.value
            output = QuantTensor(output_value)
        return output

    def __sub__(self, other):
        return self.__add__(- other)

    def __truediv__(self, other):  # todo zero point
        QuantTensor.check_input_type(other)
        if self.is_valid and other.is_valid:
            output_tensor = self.value / other.tensor
            output_scale = self.scale / other.scale
            output_bit_width = self.bit_width - other.bit_width
            output_signed = self.signed or other.signed
            output = QuantTensor(output_tensor, output_scale, output_bit_width, output_signed)
        else:
            output_value = self.value / other.value
            output = QuantTensor(output_value)
        return output

    def __abs__(self):
        if self.signed:
            return QuantTensor(
                torch.abs(self.tensor), self.zero_point, self.scale, self.bit_width - 1, False)
        else:
            return QuantTensor(
                torch.abs(self.tensor), self.zero_point, self.scale, self.bit_width, False)

    def __pos__(self):
        return self