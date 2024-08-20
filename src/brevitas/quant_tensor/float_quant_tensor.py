# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import FloatQuantTensorBase
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.torch_utils import float_internal_scale

from .float_torch_handler import FLOAT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER

IS_VALID_ATOL = 2e-1
BFLOAT16_IS_VALID_ATOL = 0.5


class FloatQuantTensor(FloatQuantTensorBase, QuantTensor):

    def __new__(
            cls,
            value,
            scale,
            zero_point,
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

    @property
    def eps(self):
        return torch.finfo(self.scale.dtype).tiny

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in FLOAT_QUANT_TENSOR_FN_HANDLER:
            return FLOAT_QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
        elif func in QUANT_TENSOR_FN_HANDLER:
            return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
        else:
            args = _unpack_quant_tensor(args)
            kwargs = _unpack_quant_tensor(kwargs)
            return func(*args, **kwargs)

    @property
    def tensor(self):
        return self.value

    @property
    def _pre_round_float_value(self):
        value = self.value
        scale = self.scale
        if self.scale.dtype == torch.bfloat16:
            value = self.value.type(torch.float32)
            scale = self.scale.type(torch.float32)
        minifloat_value = value / scale
        fp_internal_scale = 1. - self.exponent_bias - self.mantissa_bit_width
        int_scale = float_internal_scale(
            self.value, self.mantissa_bit_width, fp_internal_scale, self.eps)
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
        value_device = self.value.device
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
            int_scale = float_internal_scale(
                self.value, self.mantissa_bit_width, fp_internal_scale, self.eps)
            float_value = torch.round(self._pre_round_float_value) * int_scale
            return float_value.type(self.scale.dtype)
        else:
            raise RuntimeError(f"FloatQuantTensor not valid.")

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, FloatQuantTensor):
            raise RuntimeError("Tensor is not a FloatQuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        FloatQuantTensor.check_input_type(tensor)
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
        return self.set(value=self.value.view(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.set(value=self.value.reshape(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return self.set(value=self.value.flatten(*args, **kwargs))

    def transpose(self, *args, **kwargs):
        value = self.value.transpose(*args, **kwargs)
        tensor_meta = {'scale': self.scale, 'zero_point': self.zero_point}
        for k, tm in tensor_meta.items():
            if len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.transpose(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    def permute(self, *args, **kwargs):
        value = self.value.permute(*args, **kwargs)
        tensor_meta = {'scale': self.scale, 'zero_point': self.zero_point}
        for k, tm in tensor_meta.items():
            if len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.permute(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    @staticmethod
    def cat(tensors, dim, out=None):
        if out is not None:
            raise RuntimeError("Out not supported.")
        if len(tensors) < 2:
            return tensors[0]
        else:
            first_qt = tensors[0]
            if all([isinstance(qt, FloatQuantTensor) for qt in tensors]):
                for qt in tensors[1:]:
                    first_qt.check_scaling_factors_same(qt)
                    first_qt.check_zero_points_same(qt)
                    first_qt.check_bit_width_same(qt)
                    first_qt.check_exponent_bias(qt)
                    first_qt.check_inf_nan_same(qt)
                    first_qt.check_sign_same(qt)
                output_value = torch.cat([qt.value for qt in tensors], dim=dim)
                output_training = any([qt.training for qt in tensors])
                if output_training:
                    output_scale = sum([qt.scale for qt in tensors]) / len(tensors)
                    output_zero_point = sum([qt.zero_point for qt in tensors]) / len(tensors)
                    output_exponent_bit_width = sum([qt.exponent_bit_width for qt in tensors
                                                    ]) / len(tensors)
                    output_mantissa_bit_width = sum([qt.mantissa_bit_width for qt in tensors
                                                    ]) / len(tensors)
                    output_exponent_bias = sum([qt.exponent_bias for qt in tensors]) / len(tensors)
                else:  # at eval time, they are the same
                    output_scale = first_qt.scale
                    output_zero_point = first_qt.zero_point
                    output_exponent_bit_width = first_qt.exponent_bit_width
                    output_mantissa_bit_width = first_qt.mantissa_bit_width
                    output_exponent_bias = first_qt.exponent_bias
                output_signed = first_qt.signed  # they are the same
                output_saturating = first_qt.saturating  # they are the same
                output_inf_values = first_qt.inf_values  # they are the same
                output_nan_values = first_qt.nan_values  # they are the same
                return FloatQuantTensor(
                    value=output_value,
                    scale=output_scale,
                    zero_point=output_zero_point,
                    exponent_bit_width=output_exponent_bit_width,
                    mantissa_bit_width=output_mantissa_bit_width,
                    exponent_bias=output_exponent_bias,
                    saturating=output_saturating,
                    inf_values=output_inf_values,
                    nan_values=output_nan_values,
                    signed=output_signed,
                    training=output_training)
            else:
                tensors = [_unpack_quant_tensor(qt) for qt in tensors]
                output_value = torch.cat(tensors, dim=dim)
                return output_value

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_value = (-self.minifloat(float_datatype=True) - self.zero_point) * self.scale
        # In case the dtype of self.minifloat is different from the one of the scale
        neg_value = neg_value.type(self.scale.dtype)
        if self.signed:
            return FloatQuantTensor(
                value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
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
        return f"FloatQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, exponent_bit_width={self.exponent_bit_width}, mantissa_bit_width={self.mantissa_bit_width}, exponent_bias={self.exponent_bias}, inf_values={self.inf_values}, nan_values={self.nan_values}, signed_t={self.signed_t}, training_t={self.training_t})"

    def __truediv__(self, other):
        if isinstance(other, QuantTensor):
            return self.value / other.value
        else:
            output = self.value / other
        return output

    def __abs__(self):
        if self.signed:
            abs_value = (
                torch.abs(self.minifloat(float_datatype=True)) - self.zero_point) * self.scale
            # In case the dtype of self.minifloat is different from the one of the scale
            abs_value = abs_value.type(self.scale.dtype)
            return FloatQuantTensor(
                value=abs_value,
                scale=self.scale,
                zero_point=self.zero_point,
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
