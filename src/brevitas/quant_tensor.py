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
    def is_not_none(self):
        return (self.value is not None
                and self.scale is not None
                and self.zero_point is not None
                and self.bit_width is not None
                and self.signed is not None)

    @property
    def is_valid(self):
        if self.is_not_none:
            with torch.no_grad():
                int_value = (self.value / self.scale) + self.zero_point
                rounded_int_value = torch.round(int_value)
                is_int = torch.isclose(int_value, rounded_int_value).all()
                if self.bit_width >= 2:
                    if self.signed:
                        is_upper_b = (2.0 ** (self.bit_width - 1) - 1 >= rounded_int_value).all()
                        is_lower_b = (- 2.0 ** (self.bit_width - 1) <= rounded_int_value).all()
                    else:
                        is_upper_b = (2.0 ** (self.bit_width - 1) >= rounded_int_value).all()
                        is_lower_b = (0. <= int_value).all()
                    return (is_int & is_upper_b & is_lower_b).item()
                else:  # binary case
                    unique_vals = rounded_int_value.unique(
                        sorted=False, return_counts=False, return_inverse=False)
                    is_binary = unique_vals.view(-1).size()[0] == 2
                    return is_int.item() and is_binary
        else:
            return False

    def set(self, **kwargs):
        return self._replace(**kwargs)

    def detach_(self):
        self.value.detach_()
        self.scale.detach_()
        self.zero_point.detach_()
        self.bit_width.detach_()

    def detach(self):
        return QuantTensor(
            self.value.detach(),
            self.scale.detach() if self.scale is not None else None,
            self.zero_point.detach() if self.zero_point is not None else None,
            self.bit_width.detach() if self.bit_width is not None else None,
            self.signed,
            self.training)

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
    def check_input_type(tensor):
        if not isinstance(tensor, QuantTensor):
            raise RuntimeError("Tensor is not a QuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        QuantTensor.check_input_type(tensor)
        if tensor.zero_point is not None:
            return (tensor.zero_point != 0.).any()
        else:
            return None

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
            output_signed = first_qt.signed  # they are the same
            output_training = any([qt.training for qt in tensor_list])
            return QuantTensor(
                value=output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output_value = torch.cat([qt.value for qt in tensor_list], dim=dim)
            return QuantTensor(output_value)

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_value = (- self.int(float_datatype=True) - self.zero_point) * self.scale
        if self.signed:
            return QuantTensor(
                value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training)
        else:
            return QuantTensor(
                value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width + 1,
                signed=True,
                training=self.training)

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
            output_training = self.training or other.training
            output = QuantTensor(
                value=output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output_value = self.value + other.value
            output = QuantTensor(output_value)
        return output

    def __mul__(self, other):
        QuantTensor.check_input_type(other)
        if self.is_valid and other.is_valid:
            output_value = self.value * other.value
            output_scale = self.scale * other.scale
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point
            else:
                output_zero_point = None # TODO non-zero zero point
            output = QuantTensor(
                value=output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output_value = self.value * other.value
            output = QuantTensor(output_value)
        return output

    def __sub__(self, other):
        return self.__add__(- other)

    def __truediv__(self, other):
        QuantTensor.check_input_type(other)
        if self.is_valid and other.is_valid:
            output_tensor = self.value / other.tensor
            output_scale = self.scale / other.scale
            output_bit_width = self.bit_width - other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point / other.zero_point
            else:
                output_zero_point = None # TODO non-zero zero point
            output = QuantTensor(
                value=output_tensor,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output_value = self.value / other.value
            output = QuantTensor(output_value)
        return output

    def __abs__(self):
        if self.signed:
            abs_value = (torch.abs(self.int(float_datatype=True)) - self.zero_point) * self.scale
            return QuantTensor(
                value=abs_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width - 1,
                signed=False,
                training=self.training)
        else:
            return self

    def __pos__(self):
        return self