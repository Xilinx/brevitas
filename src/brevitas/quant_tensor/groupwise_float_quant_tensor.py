# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor.base_quant_tensor import GroupwiseFloatQuantTensorBase
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas.utils.torch_utils import float_internal_scale

from .float_torch_handler import FLOAT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER

IS_VALID_ATOL = 2e-1
BFLOAT16_IS_VALID_ATOL = 0.5


class GroupwiseFloatQuantTensor(GroupwiseFloatQuantTensorBase, QuantTensor):

    def __new__(
            cls,
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
            training):

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if not isinstance(zero_point, torch.Tensor):
            zero_point = torch.tensor(zero_point, dtype=torch.float)
        if not isinstance(exponent_bit_width, torch.Tensor):
            exponent_bit_width = torch.tensor(exponent_bit_width, dtype=torch.float)
        if not isinstance(mantissa_bit_width, torch.Tensor):
            mantissa_bit_width = torch.tensor(mantissa_bit_width, dtype=torch.float)
        if not isinstance(exponent_bias, torch.Tensor):
            exponent_bias = torch.tensor(exponent_bias, dtype=torch.float)
        if not isinstance(saturating, torch.Tensor):
            saturating = torch.tensor(saturating, dtype=torch.bool)
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
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            signed,
            training)
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
    def _pre_round_float_value(self):
        value, scale, zp = self.expand()
        if self.scale.dtype == torch.bfloat16:
            value = value.type(torch.float32)
            scale = scale.type(torch.float32)
        minifloat_value = value / scale
        fp_internal_scale = 1. - self.exponent_bias - self.mantissa_bit_width
        int_scale = float_internal_scale(self.value, self.mantissa_bit_width, fp_internal_scale)
        minifloat_value = minifloat_value / int_scale
        return minifloat_value

    @property
    def is_valid(self):
        with torch.no_grad():
            pre_round_minifloat_value = self._pre_round_float_value
            rounded_minifloat_value = torch.round(pre_round_minifloat_value)
            max_abs_diff = torch.max(torch.abs(pre_round_minifloat_value - rounded_minifloat_value))
            atol = BFLOAT16_IS_VALID_ATOL if self.value.dtype == torch.bfloat16 else IS_VALID_ATOL
            is_minifloat = max_abs_diff < atol
            # We are missing the checks about self being contained between max and min value
            # given by mantissa, exponent, inf, nan, and saturating
            return is_minifloat

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

    def minifloat(self, float_datatype=True):
        # TODO: Check if OCP and cast to proper data-type if matching
        assert float_datatype, "Minifloat quant returns only higher precision dtype"

        if self.is_valid:
            fp_internal_scale = 1. - self.exponent_bias - self.mantissa_bit_width
            int_scale = float_internal_scale(self.value, self.mantissa_bit_width, fp_internal_scale)
            float_value = torch.round(self._pre_round_float_value) * int_scale
            return float_value.type(self.scale.dtype)
        else:
            raise RuntimeError(f"FloatQuantTensor not valid.")

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, GroupwiseFloatQuantTensor):
            raise RuntimeError("Tensor is not a GroupwiseFloatQuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        GroupwiseFloatQuantTensor.check_input_type(tensor)
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
        neg_value = GroupwiseFloatQuantTensor.from_expanded(
            neg_value, self.group_size, self.group_dim, compress=False)
        scale = GroupwiseFloatQuantTensor.from_expanded(
            scale, self.group_size, self.group_dim, compress=True)
        if self.signed:
            return GroupwiseFloatQuantTensor(
                value=neg_value,
                scale=scale,
                zero_point=self.zero_point,
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
        return f"GroupwiseFloatQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, group_size={self.group_size}, group_dim={self.group_dim}, exponent_bit_width={self.exponent_bit_width}, mantissa_bit_width={self.mantissa_bit_width}, exponent_bias={self.exponent_bias}, inf_values={self.inf_values}, nan_values={self.nan_values}, signed_t={self.signed_t}, training_t={self.training_t})"

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
            abs_value = GroupwiseFloatQuantTensor.from_expanded(
                abs_value, self.group_size, self.group_dim, compress=False)
            scale = GroupwiseFloatQuantTensor.from_expanded(
                scale, self.group_size, self.group_dim, compress=True)
            return GroupwiseFloatQuantTensor(
                value=abs_value,
                scale=self.scale,
                zero_point=self.zero_point,
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
