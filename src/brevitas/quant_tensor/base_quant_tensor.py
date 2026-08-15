# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from brevitas.function.ops_ste import round_ste
from brevitas.utils.torch_utils import float_internal_scale

TOLERANCE = {torch.float64: 1e-1, torch.float32: 2e-1, torch.float16: 0.5, torch.bfloat16: 0.5}


# Base class for all QuantTensor types.
# Subclasses torch.Tensor, where the underlying tensor data represents the dequantized `value`.
# Quantization metadata (scale, zero_point, bit_width, etc.) are stored as regular attributes.
class QuantTensor(Tensor):

    @staticmethod
    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float)
        # Create tensor subclass wrapping the value data.
        # Use as_subclass to preserve grad_fn and requires_grad.
        return value.as_subclass(cls)

    def __init__(self, value, *args, **kwargs):
        # Subclasses should NOT call super().__init__() with args;
        # metadata is set by subclass __init__ methods.
        pass

    @property
    def value(self):
        # The tensor itself IS the value.
        # Return a plain Tensor view to avoid infinite recursion in operations.
        # Use as_subclass to preserve grad_fn and requires_grad.
        return Tensor.as_subclass(self, Tensor)

    # Mapping from _fields names to constructor parameter names.
    # Override in subclasses where they differ (e.g. groupwise: scale_ -> scale).
    _field_to_constructor_param = {}

    def _get_constructor_kwargs(self):
        """Return a dict of constructor_param_name -> value for all metadata fields."""
        result = {}
        for field in self._fields:
            param_name = self._field_to_constructor_param.get(field, field)
            result[param_name] = getattr(self, field)
        return result

    def set(self, **kwargs):
        """Create a new QuantTensor with some fields replaced.
        Equivalent to NamedTuple._replace()."""
        value = kwargs.pop('value', None)
        value_ = kwargs.pop('value_', None)
        ctor_kwargs = self._get_constructor_kwargs()
        # Map any field-name kwargs to constructor param names
        mapped_kwargs = {}
        for k, v in kwargs.items():
            param_name = self._field_to_constructor_param.get(k, k)
            mapped_kwargs[param_name] = v
        ctor_kwargs.update(mapped_kwargs)
        # Remove value/value_ from kwargs dict (they're positional)
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        if value is not None:
            new_value = value
        elif value_ is not None:
            new_value = value_
        else:
            if hasattr(self, '_is_groupwise') and self._is_groupwise:
                new_value = self._value_
            else:
                new_value = self.value
        return type(self)(new_value, **ctor_kwargs)

    def detach_(self):
        super().detach_()
        for field in self._fields:
            val = getattr(self, field)
            if isinstance(val, Tensor):
                val.detach_()

    def detach(self):
        ctor_kwargs = self._get_constructor_kwargs()
        for k, v in ctor_kwargs.items():
            if isinstance(v, Tensor):
                ctor_kwargs[k] = v.detach()
        new_value = self.value.detach() if not (
            hasattr(self, '_is_groupwise') and self._is_groupwise) else self._value_.detach()
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        return type(self)(new_value, **ctor_kwargs)

    def contiguous(self):
        ctor_kwargs = self._get_constructor_kwargs()
        for k, v in ctor_kwargs.items():
            if isinstance(v, Tensor):
                ctor_kwargs[k] = v.contiguous()
        new_value = self.value.contiguous() if not (
            hasattr(self, '_is_groupwise') and self._is_groupwise) else self._value_.contiguous()
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        return type(self)(new_value, **ctor_kwargs)

    def __getitem__(self, index):
        """Index the leading dimension while preserving aligned quantization metadata."""
        if isinstance(index, Tensor):
            if index.dim() != 0 or index.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError('QuantTensor indices must be integer scalars.')
            index = int(index.item())
        if not isinstance(index, (int, slice)):
            raise TypeError('QuantTensor indexing supports an integer or slice.')

        is_groupwise = getattr(self, '_is_groupwise', False)
        source_value = self._value_ if is_groupwise else self.value
        original_shape = self.value.shape
        ctor_kwargs = self._get_constructor_kwargs()
        for name, metadata in tuple(ctor_kwargs.items()):
            if isinstance(metadata,
                          Tensor) and metadata.dim() > 0 and metadata.shape[0] == original_shape[0]:
                ctor_kwargs[name] = metadata[index]

        if is_groupwise:
            if isinstance(index, int):
                group_dim = self.group_dim
                normalized_group_dim = group_dim if group_dim >= 0 else group_dim + len(
                    original_shape)
                if normalized_group_dim == 0:
                    raise RuntimeError('Cannot remove the grouped dimension through indexing.')
                ctor_kwargs['group_dim'] = group_dim - 1 if group_dim > 0 else group_dim
                if self.dequant_shape is not None:
                    ctor_kwargs['dequant_shape'] = tuple(self.dequant_shape[1:])
            elif self.dequant_shape is not None:
                ctor_kwargs['dequant_shape'] = tuple(self.value[index].shape)
        return type(self)(source_value[index], **ctor_kwargs)

    @property
    def shape(self):
        return self.value.shape

    def dim(self):
        return self.value.dim()

    def add(self, other):
        return self + other

    def to(self, *args, **kwargs):
        ctor_kwargs = self._get_constructor_kwargs()
        for k, v in ctor_kwargs.items():
            if isinstance(v, Tensor):
                ctor_kwargs[k] = v.to(*args, **kwargs)
        new_value = self.value.to(
            *args, **kwargs) if not (hasattr(self, '_is_groupwise') and
                                     self._is_groupwise) else self._value_.to(*args, **kwargs)
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        return type(self)(new_value, **ctor_kwargs)

    def cuda(self, *args, **kwargs):
        ctor_kwargs = self._get_constructor_kwargs()
        for k, v in ctor_kwargs.items():
            if isinstance(v, Tensor):
                ctor_kwargs[k] = v.cuda(*args, **kwargs)
        new_value = self.value.cuda(
            *args, **kwargs) if not (hasattr(self, '_is_groupwise') and
                                     self._is_groupwise) else self._value_.cuda(*args, **kwargs)
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        return type(self)(new_value, **ctor_kwargs)

    def cpu(self, *args, **kwargs):
        ctor_kwargs = self._get_constructor_kwargs()
        for k, v in ctor_kwargs.items():
            if isinstance(v, Tensor):
                ctor_kwargs[k] = v.cpu(*args, **kwargs)
        new_value = self.value.cpu(
            *args, **kwargs) if not (hasattr(self, '_is_groupwise') and
                                     self._is_groupwise) else self._value_.cpu(*args, **kwargs)
        ctor_kwargs.pop('value', None)
        ctor_kwargs.pop('value_', None)
        return type(self)(new_value, **ctor_kwargs)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __pos__(self):
        return self

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)

    @staticmethod
    def is_zero_zero_point(tensor):
        return (tensor.zero_point == 0.).all()


class IntMixin:

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
            atol = TOLERANCE[self.value.dtype]
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
            raise RuntimeError(f"QuantTensor not valid.")

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


class FloatMixin:

    @property
    def _pre_round_float_value(self):
        value = self.value
        scale = self.scale
        if self.scale.dtype == torch.bfloat16:
            value = self.value.type(torch.float32)
            scale = self.scale.type(torch.float32)
        minifloat_value = value / scale
        fp_internal_scale = 1. - self.exponent_bias - self.mantissa_bit_width
        eps = torch.finfo(scale.dtype).tiny
        int_scale = float_internal_scale(
            minifloat_value, self.mantissa_bit_width, fp_internal_scale, eps)
        minifloat_value = minifloat_value / int_scale
        return minifloat_value

    @property
    def is_valid(self):
        with torch.no_grad():
            pre_round_minifloat_value = self._pre_round_float_value
            rounded_minifloat_value = torch.round(pre_round_minifloat_value)
            max_abs_diff = torch.max(torch.abs(pre_round_minifloat_value - rounded_minifloat_value))
            atol = TOLERANCE[self.value.dtype]
            is_minifloat = max_abs_diff < atol
            # We are missing the checks about self being contained between max and min value
            # given by mantissa, exponent, inf, nan, and saturating
            return is_minifloat

    def minifloat(self, float_datatype=True):
        # TODO: Check if OCP and cast to proper data-type if matching
        assert float_datatype, "Minifloat quant returns only higher precision dtype"
        if self.is_valid:
            value = self.value
            scale = self.scale
            if self.scale.dtype == torch.bfloat16:
                value = self.value.type(torch.float32)
                scale = self.scale.type(torch.float32)
            minifloat_value = value / scale
            fp_internal_scale = 1. - self.exponent_bias - self.mantissa_bit_width
            eps = torch.finfo(scale.dtype).tiny
            int_scale = float_internal_scale(
                minifloat_value, self.mantissa_bit_width, fp_internal_scale, eps)
            float_value = torch.round(self._pre_round_float_value) * int_scale
            return float_value.type(self.scale.dtype)
        else:
            raise RuntimeError(f"FloatQuantTensor not valid.")

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

    def int(self):
        value = torch.round(self._pre_round_float_value)
        return value


def _unpack_quant_tensor(input_data):
    if isinstance(input_data, QuantTensor):
        return input_data.value
    elif isinstance(input_data, tuple):
        return tuple([_unpack_quant_tensor(v) for v in input_data])
    elif isinstance(input_data, list):
        return [_unpack_quant_tensor(v) for v in input_data]
    elif isinstance(input_data, dict):
        return {k: _unpack_quant_tensor(v) for k, v in input_data.items()}
    else:
        return input_data
