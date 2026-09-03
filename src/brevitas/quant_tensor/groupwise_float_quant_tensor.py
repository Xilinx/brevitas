# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.quant_tensor.base_quant_tensor import FloatMixin
from brevitas.quant_tensor.base_quant_tensor import GroupwiseQuantTensorMixin
from brevitas.quant_tensor.base_quant_tensor import QuantTensor


class GroupwiseFloatQuantTensor(GroupwiseQuantTensorMixin, FloatMixin, QuantTensor):

    _constructor_metadata = {
        'scale': '_scale',
        'zero_point': '_zero_point',
        'group_size': '_group_size',
        'group_dim': '_group_dim',
        'exponent_bit_width': '_exponent_bit_width',
        'mantissa_bit_width': '_mantissa_bit_width',
        'exponent_bias': '_exponent_bias',
        'saturating': '_saturating',
        'inf_values': '_inf_values',
        'nan_values': '_nan_values',
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
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            signed,
            training,
            dequant_shape=None):
        device = self._value.device
        scale = self._as_tensor(scale, torch.float, device)
        zero_point = self._as_tensor(zero_point, torch.float, device)
        exponent_bit_width = self._as_tensor(exponent_bit_width, torch.float, device)
        mantissa_bit_width = self._as_tensor(mantissa_bit_width, torch.float, device)
        exponent_bias = self._as_tensor(exponent_bias, torch.float, device)
        saturating = self._as_tensor(saturating, torch.bool, device)
        signed = self._as_tensor(signed, torch.bool, device)
        training = self._as_tensor(training, torch.bool, device)
        self._exponent_bit_width = exponent_bit_width
        self._mantissa_bit_width = mantissa_bit_width
        self._exponent_bias = exponent_bias
        self._saturating = saturating
        self._inf_values = inf_values
        self._nan_values = nan_values
        self._set_groupwise_metadata(
            scale, zero_point, group_size, group_dim, signed, training, dequant_shape)

    @property
    def exponent_bit_width(self):
        return self._exponent_bit_width

    @exponent_bit_width.setter
    def exponent_bit_width(self, value):
        self._exponent_bit_width = value

    @property
    def mantissa_bit_width(self):
        return self._mantissa_bit_width

    @mantissa_bit_width.setter
    def mantissa_bit_width(self, value):
        self._mantissa_bit_width = value

    @property
    def exponent_bias(self):
        return self._exponent_bias

    @exponent_bias.setter
    def exponent_bias(self, value):
        self._exponent_bias = value

    @property
    def inf_values(self):
        return self._inf_values

    @inf_values.setter
    def inf_values(self, value):
        self._inf_values = value

    @property
    def nan_values(self):
        return self._nan_values

    @nan_values.setter
    def nan_values(self, value):
        self._nan_values = value

    @property
    def saturating(self):
        return self._saturating.item()

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_deq = -self.minifloat(float_datatype=True)
        _, scale, zp = self.expand()

        neg_value = (-neg_deq - zp) * scale
        # In case the dtype of self.minifloat is different from the one of the scale
        neg_value = neg_value.type(scale.dtype)
        neg_value = GroupwiseFloatQuantTensor.from_expanded(
            neg_value, self.group_size, self.group_dim, compress=False)
        scale = GroupwiseFloatQuantTensor.from_expanded(
            scale, self.group_size, self.group_dim, compress=True)
        if self.signed:
            return GroupwiseFloatQuantTensor(
                value=neg_value,
                scale=scale,
                zero_point=self._zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                exponent_bit_width=self.exponent_bit_width,
                mantissa_bit_width=self.mantissa_bit_width,
                exponent_bias=self.exponent_bias,
                saturating=self.saturating,
                inf_values=self.inf_values,
                nan_values=self.nan_values,
                signed=self.signed,
                training=self.training)
        else:
            # TODO: implement
            raise NotImplementedError

    def __str__(self):
        return f"GroupwiseFloatQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, group_size={self.group_size}, group_dim={self.group_dim}, exponent_bit_width={self.exponent_bit_width}, mantissa_bit_width={self.mantissa_bit_width}, exponent_bias={self.exponent_bias}, inf_values={self.inf_values}, nan_values={self.nan_values}, signed={self._signed}, training={self._training})"

    def __abs__(self):
        if self.signed:
            neg_deq = self.minifloat(float_datatype=True)
            _, scale, zp = self.expand()

            # In case the dtype of self.minifloat is different from the one of the scale
            abs_value = (neg_deq - zp) * scale
            # In case the dtype of self.minifloat is different from the one of the scale
            abs_value = abs_value.type(scale.dtype)
            abs_value = GroupwiseFloatQuantTensor.from_expanded(
                abs_value, self.group_size, self.group_dim, compress=False)
            scale = GroupwiseFloatQuantTensor.from_expanded(
                scale, self.group_size, self.group_dim, compress=True)
            return GroupwiseFloatQuantTensor(
                value=abs_value,
                scale=self._scale,
                zero_point=self._zero_point,
                group_size=self.group_size,
                group_dim=self.group_dim,
                exponent_bit_width=self.exponent_bit_width,
                mantissa_bit_width=self.mantissa_bit_width,
                exponent_bias=self.exponent_bias,
                saturating=self.saturating,
                inf_values=self.inf_values,
                nan_values=self.nan_values,
                signed=False,
                training=self.training)
        else:
            return self
