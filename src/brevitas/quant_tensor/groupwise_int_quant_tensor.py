# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor.base_quant_tensor import GroupwisIntQuantTensorBase
from brevitas.quant_tensor.base_quant_tensor import IntMixin
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas.utils.torch_utils import float_internal_scale

from .int_torch_handler import INT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER


class GroupwiseIntQuantTensor(GroupwisIntQuantTensorBase, IntMixin, QuantTensor):

    def __new__(
            cls,
            value,
            scale,
            zero_point,
            group_size,
            group_dim,
            bit_width,
            signed,
            training,
            dequant_shape=None):

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
            cls,
            value,
            scale,
            zero_point,
            group_size,
            group_dim,
            bit_width,
            signed,
            training,
            dequant_shape)
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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in QUANT_TENSOR_FN_HANDLER:
            return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
        else:
            args = _unpack_quant_tensor(args)
            kwargs = _unpack_quant_tensor(kwargs)
            return func(*args, **kwargs)

    def expand(self):
        from brevitas.utils.quant_utils import groupwise_dequant_expand
        return groupwise_dequant_expand(
            self.value_, self.scale_, self.zero_point_, self.group_dim, self.dequant_shape)

    @staticmethod
    def from_expanded(value, group_size, group_dim, compress=False):
        group_dim = group_dim if group_dim >= 0 else group_dim - 1
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

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, GroupwiseIntQuantTensor):
            raise RuntimeError("Tensor is not a GroupwiseIntQuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        GroupwiseIntQuantTensor.check_input_type(tensor)
        return (tensor.zero_point == 0.).all()

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
