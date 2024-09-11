# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops_ste import round_ste
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import IntQuantTensorBase
from brevitas.quant_tensor import QuantTensor

from .int_torch_handler import INT_QUANT_TENSOR_FN_HANDLER
from .torch_handler import QUANT_TENSOR_FN_HANDLER

IS_VALID_ATOL = 2e-1
B_FLOAT16_IS_VALID_ATOL = 0.5


class IntQuantTensor(IntQuantTensorBase, QuantTensor):

    def __new__(cls, value, scale, zero_point, bit_width, signed, training):

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
        quant_tensor = super().__new__(cls, value, scale, zero_point, bit_width, signed, training)
        return quant_tensor

    @property
    def signed(self):
        return self.signed_t.item()

    @property
    def training(self):
        return self.training_t.item()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in INT_QUANT_TENSOR_FN_HANDLER:
            return INT_QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
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
            atol = B_FLOAT16_IS_VALID_ATOL if self.value.dtype in (
                torch.bfloat16, torch.float16) else IS_VALID_ATOL
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
        value_device = self.value.device
        is_same_device = True
        for t in [self.scale, self.zero_point, self.bit_width]:
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
            raise RuntimeError(f"IntQuantTensor not valid.")

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, IntQuantTensor):
            raise RuntimeError("Tensor is not a IntQuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        IntQuantTensor.check_input_type(tensor)
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
        if not torch.allclose(self.bit_width, other.bit_width):
            raise RuntimeError("Bit widths are different")

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
        tensor_meta = {
            'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.transpose(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    def permute(self, *args, **kwargs):
        value = self.value.permute(*args, **kwargs)
        tensor_meta = {
            'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if len(value.shape) == len(tm.shape):
                tensor_meta[k] = tm.permute(*args, **kwargs)
        return self.set(value=value, **tensor_meta)

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)

    @property
    def ndim(self):
        return self.value.ndim

    def dim(self):
        return self.value.dim()

    @property
    def shape(self):
        return self.value.shape

    def dim(self):
        return self.value.dim()

    def add(self, other):
        return self + other

    @staticmethod
    def cat(tensors, dim, out=None):
        if out is not None:
            raise RuntimeError("Out not supported.")
        if len(tensors) < 2:
            return tensors[0]
        else:
            first_qt = tensors[0]
            if all([isinstance(qt, IntQuantTensor) for qt in tensors]):
                for qt in tensors[1:]:
                    first_qt.check_scaling_factors_same(qt)
                    first_qt.check_zero_points_same(qt)
                    first_qt.check_bit_width_same(qt)
                    first_qt.check_sign_same(qt)
                output_value = torch.cat([qt.value for qt in tensors], dim=dim)
                output_training = any([qt.training for qt in tensors])
                if output_training:
                    output_scale = sum([qt.scale for qt in tensors]) / len(tensors)
                    output_zero_point = sum([qt.zero_point for qt in tensors]) / len(tensors)
                    output_bit_width = sum([qt.bit_width for qt in tensors]) / len(tensors)
                else:  # at eval time, they are the same
                    output_scale = first_qt.scale
                    output_zero_point = first_qt.zero_point
                    output_bit_width = first_qt.bit_width
                output_signed = first_qt.signed  # they are the same
                return IntQuantTensor(
                    value=output_value,
                    scale=output_scale,
                    zero_point=output_zero_point,
                    bit_width=output_bit_width,
                    signed=output_signed,
                    training=output_training)
            else:
                tensors = [_unpack_quant_tensor(qt) for qt in tensors]
                output_value = torch.cat(tensors, dim=dim)
                return output_value

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_value = (-self.int(float_datatype=True) - self.zero_point) * self.scale
        # In case the dtype of self.int is different from the one of the scale
        neg_value = neg_value.type(self.scale.dtype)
        if self.signed:
            return IntQuantTensor(
                value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training)
        else:
            return IntQuantTensor(
                value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width + 1,
                signed=True,
                training=self.training)

    def __add__(self, other):
        if isinstance(other, IntQuantTensor):
            self.check_scaling_factors_same(other)
            output_value = self.value + other.value
            output_scale = (self.scale + other.scale) / 2
            output_zero_point = self.zero_point + other.zero_point
            max_val = max_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            max_val += max_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            min_val = min_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            min_val += min_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            output_bit_width = ceil_ste(torch.log2(max_val - min_val))
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            output = IntQuantTensor(
                value=output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        elif isinstance(other, QuantTensor):
            output = self.value + _unpack_quant_tensor(other)
        else:
            # When adding a QT with a normal Tensor, we use the zero_point as a way to preserve
            # and return a QT.
            output = IntQuantTensor(
                value=self.value + _unpack_quant_tensor(other),
                scale=self.scale,
                zero_point=self.zero_point - _unpack_quant_tensor(other) / self.scale,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training)
        return output

    def __mul__(self, other):
        if isinstance(other, IntQuantTensor):
            output_value = self.value * other.value
            output_scale = self.scale * other.scale
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point
            else:
                raise RuntimeError("Zero-points of mul operands are non-zero, not supported.")
            output = IntQuantTensor(
                value=output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output = self.value * _unpack_quant_tensor(other)
        return output

    def __str__(self):
        return f"IntQuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, bit_width={self.bit_width}, signed_t={self.signed_t}, training_t={self.training_t})"

    def __truediv__(self, other):
        if isinstance(other, IntQuantTensor):
            output_tensor = self.value / other.value  # Note, output tensor not guaranteed to pass self.is_valid()
            max_int_denominator = 2 ** (other.bit_width - int(other.signed))
            output_scale = self.scale / (other.scale * max_int_denominator)
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point  # Output zero_point is a new, zero-valued tensor
            else:
                raise RuntimeError("Zero-points of div operands are non-zero, not supported.")
            output = IntQuantTensor(
                value=output_tensor,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output = self.value / _unpack_quant_tensor(other)
        return output

    def __abs__(self):
        if self.signed:
            abs_value = (torch.abs(self.int(float_datatype=True)) - self.zero_point) * self.scale
            # In case the dtype of self.int is different from the one of the scale
            abs_value = abs_value.type(self.scale.dtype)
            return IntQuantTensor(
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
