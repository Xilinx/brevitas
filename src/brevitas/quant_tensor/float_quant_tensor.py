# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import FloatQuantTensorBase
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor.base_quant_tensor import FloatMixin

from .float_torch_handler import FLOAT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER


class FloatQuantTensor(FloatQuantTensorBase, FloatMixin, QuantTensor):

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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
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

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, FloatQuantTensor):
            raise RuntimeError("Tensor is not a FloatQuantTensor")

    # Magic methods can't live in the Mixin class
    def __add__(self, other):
        if isinstance(other, QuantTensor):
            return self.value + other.value
        else:
            return self.value + other

    def __mul__(self, other):
        if isinstance(other, QuantTensor):
            return self.value * other.value
        else:
            return self.value * other

    def __truediv__(self, other):
        if isinstance(other, QuantTensor):
            return self.value / other.value
        else:
            return self.value / other

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

    def __str__(self):
        return f"FloatQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, exponent_bit_width={self.exponent_bit_width}, mantissa_bit_width={self.mantissa_bit_width}, exponent_bias={self.exponent_bias}, inf_values={self.inf_values}, nan_values={self.nan_values}, signed_t={self.signed_t}, training_t={self.training_t})"

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
