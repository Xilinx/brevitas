# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor.base_quant_tensor import GroupwiseQuantTensorMixin
from brevitas.quant_tensor.base_quant_tensor import IntMixin
from brevitas.quant_tensor.base_quant_tensor import QuantTensor


class GroupwiseIntQuantTensor(GroupwiseQuantTensorMixin, IntMixin, QuantTensor):

    _constructor_metadata = {
        'scale': '_scale',
        'zero_point': '_zero_point',
        'group_size': '_group_size',
        'group_dim': '_group_dim',
        'bit_width': '_bit_width',
        'signed': '_signed',
        'training': '_training',
        'dequant_shape': '_dequant_shape'}

    def __init__(
            self,
            value,
            scale,
            zero_point,
            group_size,
            group_dim,
            bit_width,
            signed,
            training,
            dequant_shape=None):
        scale = self._as_tensor(scale, torch.float)
        zero_point = self._as_tensor(zero_point, torch.float)
        bit_width = self._as_tensor(bit_width, torch.float)
        signed = self._as_tensor(signed, torch.bool)
        training = self._as_tensor(training, torch.bool)
        self._bit_width = bit_width
        self._set_groupwise_metadata(
            scale, zero_point, group_size, group_dim, signed, training, dequant_shape)

    @property
    def bit_width(self):
        return self._bit_width

    @bit_width.setter
    def bit_width(self, value):
        self._bit_width = value

    @staticmethod
    def is_zero_zero_point(tensor):
        GroupwiseIntQuantTensor.check_input_type(tensor)
        return (tensor.zero_point == 0.).all()

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
                zero_point=self._zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training)
        else:
            # TODO: implement
            raise NotImplementedError

    def __str__(self):
        return f"GroupwiseIntQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, group_size={self.group_size}, group_dim={self.group_dim}, bit_width={self.bit_width}, signed={self._signed}, training={self._training})"

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
                scale=self._scale,
                zero_point=self._zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                bit_width=self.bit_width,
                signed=False,
                training=self.training)
        else:
            return self
