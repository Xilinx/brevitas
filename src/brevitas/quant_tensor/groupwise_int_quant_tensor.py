# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor.base_quant_tensor import GroupwisIntQuantTensorBase
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas.utils.torch_utils import float_internal_scale

from .int_torch_handler import INT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER

IS_VALID_ATOL = 2e-1
BFLOAT16_IS_VALID_ATOL = 0.5


class GroupwiseIntQuantTensor(GroupwisIntQuantTensorBase, QuantTensor):

    def __new__(cls, value, scale, zero_point, group_size, group_dim, bit_width, signed, training):

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if not isinstance(zero_point, torch.Tensor):
            zero_point = torch.tensor(zero_point, dtype=torch.float)
        if not isinstance(bit_width, torch.Tensor):
            bit_width = torch.tensor(bit_width, dtype=torch.float)
        if not isinstance(signed, torch.Tensor):
            signed = torch.tensor(signed, dtype=torch.bool)
        if not isinstance(training, torch.Tensor):
            training = torch.tensor(training, dtype=torch.bool)
        quant_tensor = super().__new__(
            cls, value, scale, zero_point, group_size, group_dim, bit_width, signed, training)
        return quant_tensor

    @property
    def signed(self):
        return self.signed_t.item()

    @property
    def training(self):
        return self.training_t.item()

    @property
    def saturating(self):
        return self.saturating_t.item()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in QUANT_TENSOR_FN_HANDLER:
            return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
        else:
            args = _unpack_quant_tensor(args)
            kwargs = _unpack_quant_tensor(kwargs)
            return func(*args, **kwargs)

    def expand(self):
        curr_shape = self.value_.shape
        start_dim = self.group_dim if self.group_dim != -1 else -2
        new_value = self.value_.flatten(start_dim, start_dim + 1)
        if self.scale_.shape != ():
            new_scale = self.scale_.expand(curr_shape).flatten(start_dim, start_dim + 1)
        else:
            new_scale = self.scale_
        if self.zero_point_.shape != ():
            new_zp = self.zero_point_.expand(curr_shape).flatten(start_dim, start_dim + 1)
        else:
            new_zp = self.zero_point_

        return new_value, new_scale, new_zp

    @staticmethod
    def from_expanded(value, group_size, group_dim, compress=False):
        group_dim = group_dim if group_dim != -1 else -2
        size = list(value.shape)
        assert size[group_dim] % group_size == 0, 'Input channel is not divisible by group size'
        if compress:
            size[group_dim] = 1
        else:
            size[group_dim] = size[group_dim] // group_size
        size.insert(group_dim + 1, group_size)
        new_value = value.view(size)
        return new_value

    @property
    def tensor(self):
        return self.value

    @property
    def value(self):
        new_value, new_scale, new_zp = self.expand()
        return new_value

    @property
    def scale(self):
        new_value, new_scale, new_zp = self.expand()
        return new_scale

    @property
    def zero_point(self):
        new_value, new_scale, new_zp = self.expand()
        return new_zp

    @property
    def _pre_round_int_value(self):
        value = self.value
        scale = self.scale
        zero_point = self.zero_point
        if self.scale.dtype == torch.bfloat16:
            value = self.value.type(torch.float32)
            scale = self.scale.type(torch.float32)
            zero_point = self.zero_point.type(torch.float32)
        int_value = value / scale
        int_value = int_value + zero_point
        return int_value

    @property
    def is_valid(self):
        with torch.no_grad():
            pre_round_int_value = self._pre_round_int_value
            rounded_int_value = torch.round(pre_round_int_value)
            max_abs_diff = torch.max(torch.abs(pre_round_int_value - rounded_int_value))
            atol = BFLOAT16_IS_VALID_ATOL if self.value.dtype == torch.bfloat16 else IS_VALID_ATOL
            is_int = max_abs_diff < atol
            if self.bit_width >= 2:
                if self.signed:
                    is_upper_b = (2.0 ** (self.bit_width - 1) - 1 >= rounded_int_value).all()
                    is_lower_b = (-2.0 ** (self.bit_width - 1) <= rounded_int_value).all()
                else:
                    is_upper_b = (2.0 ** self.bit_width - 1 >= rounded_int_value).all()
                    is_lower_b = (0. <= rounded_int_value).all()
                return (is_int & is_upper_b & is_lower_b).item()
            else:  # binary case
                unique_vals = rounded_int_value.unique(
                    sorted=False, return_counts=False, return_inverse=False)
                is_binary = unique_vals.view(-1).size()[0] == 2
                is_signed = (unique_vals < 0.).any().item()
                sign_match = is_signed == self.signed
                return is_int.item() and is_binary and sign_match

    @property
    def device(self):
        value_device = self.value_.device
        is_same_device = True
        for t in [self.scale,
                  self.zero_point,
                  self.exponent_bit_width,
                  self.mantissa_bit_width,
                  self.exponent_bias]:
            is_same_device &= value_device == t.device
        if not is_same_device:
            raise RuntimeError("Value and metadata are on different devices")
        return value_device

    def int(self, float_datatype=False):
        if self.is_valid:
            int_value = round_ste(self._pre_round_int_value)
            if float_datatype:
                # Values at 8bit and lower can be represented exactly with float16 and bfloat16
                # otherwise (e.g. Int16 bias), we upscale to float32
                if self.bit_width <= 8.:
                    return int_value.type(self.scale.dtype)
                else:
                    return int_value.type(torch.float32)
            else:
                if self.bit_width <= 8. and self.signed_t.item():
                    return int_value.to(torch.int8)
                elif self.bit_width <= 8. and not self.signed_t.item():
                    return int_value.to(torch.uint8)
                else:
                    return int_value.to(torch.int32)
        else:
            raise RuntimeError(f"GroupwiseIntQuantTensor not valid.")

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, GroupwiseIntQuantTensor):
            raise RuntimeError("Tensor is not a GroupwiseIntQuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        GroupwiseIntQuantTensor.check_input_type(tensor)
        return (tensor.zero_point == 0.).all()

    def check_scaling_factors_same(self, other):
        if self.training:
            return True
        if not torch.allclose(self.scale, other.scale):
            raise RuntimeError("Scaling factors are different")

    def check_zero_points_same(self, other):
        if self.training:
            return True
        if not torch.allclose(self.zero_point, other.zero_point):
            raise RuntimeError("Zero points are different")

    def check_bit_width_same(self, other):
        if not torch.allclose(self.exponent_bit_width,
                              other.exponent_bit_width) and not torch.allclose(
                                  self.mantissa_bit_width, other.mantissa_bit_width):
            raise RuntimeError("Bit widths are different")

    def check_exponent_bias(self, other):
        if not torch.allclose(self.exponent_bias, other.exponent_bias):
            raise RuntimeError("Bit widths are different")

    def check_inf_nan_same(self, other):
        if not (set(self.inf_values) == set(other.inf_values)) and not (set(self.nan_values) == set(
                other.nan_values)):
            raise RuntimeError("Floating point representations are different")

    def check_sign_same(self, other):
        if not self.signed == other.signed:
            raise RuntimeError("Signs are different")

    def view(self, *args, **kwargs):
        return self.value.view(*args, **kwargs)  #self.set(value=self.value.view(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.value.reshape(
            *args, **kwargs)  # self.set(value=self.value.reshape(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return self.value.flatten(
            *args, **kwargs)  #self.set(value=self.value.flatten(*args, **kwargs))

    def transpose(self, *args, **kwargs):
        value = self.value.transpose(*args, **kwargs)
        return value

    def permute(self, *args, **kwargs):
        value = self.value.permute(*args, **kwargs)
        return value

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_deq = -self.minifloat(float_datatype=True)
        _, scale, zp = self.expand()

        neg_value = (-neg_deq - zp) * scale
        # In case the dtype of self.minifloat is different from the one of the scale
        neg_value = neg_value.type(scale.dtype)
        neg_value = GroupwiseIntQuantTensor.from_expanded(
            neg_value, self.group_size, self.group_dim, compress=False)
        scale = GroupwiseIntQuantTensor.from_expanded(
            scale, self.group_size, self.group_dim, compress=True)
        if self.signed:
            return GroupwiseIntQuantTensor(
                value=neg_value,
                scale=scale,
                zero_point=self.zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training,
                saturating=self.saturating)
        else:
            # TODO: implement
            raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, QuantTensor):
            return self.value + other.value
        else:
            output = self.value + other
        return output

    def __mul__(self, other):
        if isinstance(other, QuantTensor):
            return self.value * other.value
        else:
            output = self.value * other
        return output

    def __str__(self):
        return f"GroupwiseIntQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, group_size={self.group_size}, group_dim={self.group_dim}, bit_width={self.bit_width}, signed_t={self.signed_t}, training_t={self.training_t})"

    def __truediv__(self, other):
        if isinstance(other, QuantTensor):
            return self.value / other.value
        else:
            output = self.value / other
        return output

    def __abs__(self):
        if self.signed:
            neg_deq = self.minifloat(float_datatype=True)
            _, scale, zp = self.expand()

            # In case the dtype of self.minifloat is different from the one of the scale
            abs_value = (neg_deq - zp) * scale
            # In case the dtype of self.minifloat is different from the one of the scale
            abs_value = abs_value.type(scale.dtype)
            abs_value = GroupwiseIntQuantTensor.from_expanded(
                abs_value, self.group_size, self.group_dim, compress=False)
            scale = GroupwiseIntQuantTensor.from_expanded(
                scale, self.group_size, self.group_dim, compress=True)
            return GroupwiseIntQuantTensor(
                value=abs_value,
                scale=self.scale,
                zero_point=self.zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                bit_width=self.bit_width,
                signed=False,
                training=self.training,
                saturating=self.saturating)
        else:
            return self
