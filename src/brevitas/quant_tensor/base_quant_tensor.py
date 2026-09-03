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
_MISSING = object()


# Base class for all QuantTensor types.
# Subclasses torch.Tensor, where the underlying data is raw storage: dequantized for regular
# QuantTensors and grouped for groupwise QuantTensors.
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
    def _value(self):
        """Return the raw Tensor representation stored by this Tensor subclass."""
        return Tensor.as_subclass(self, Tensor)

    @property
    def value(self):
        # The tensor itself IS the value.
        # Return a plain Tensor view to avoid infinite recursion in operations.
        # Use as_subclass to preserve grad_fn and requires_grad.
        return self._value

    # Constructor parameter names mapped to private metadata attributes.
    _constructor_metadata = {}

    @staticmethod
    def _as_tensor(value, dtype, device):
        """Convert metadata literals without moving tensor-valued metadata."""
        return value if isinstance(value, Tensor) else torch.tensor(
            value, dtype=dtype, device=device)

    def _get_constructor_kwargs(self):
        """Return constructor metadata from its private backing attributes."""
        return {
            parameter: getattr(self, attribute) for parameter,
            attribute in self._constructor_metadata.items()}

    def _reconstruct(self, value, ctor_kwargs=None, metadata_transform=None):
        """
        Rebuild this type from constructor-form value and metadata.

        ``value`` must use the representation expected by the concrete
        constructor: dequantized for regular QuantTensors and grouped for
        groupwise QuantTensors.
        """
        if ctor_kwargs is None:
            ctor_kwargs = self._get_constructor_kwargs()
        else:
            ctor_kwargs = dict(ctor_kwargs)
        if metadata_transform is not None:
            for parameter, metadata in ctor_kwargs.items():
                if isinstance(metadata, Tensor):
                    ctor_kwargs[parameter] = metadata_transform(parameter, metadata)
        return type(self)(value, **ctor_kwargs)

    def _apply_and_reconstruct(self, tensor_op, *args, **kwargs):
        """Apply a Tensor operation to the constructor value and tensor metadata."""
        return self._reconstruct(
            tensor_op(self._value, *args, **kwargs),
            metadata_transform=lambda _,
            metadata: tensor_op(metadata, *args, **kwargs))

    def _metadata_on_device(self, device):
        """Return whether every tensor-backed metadata field is on ``device``."""
        return all(
            device == getattr(self, attribute).device
            for attribute in self._constructor_metadata.values()
            if isinstance(getattr(self, attribute), Tensor))

    def set(self, **kwargs):
        """
        Create a new QuantTensor with some fields replaced.

        ``value`` replaces the raw constructor value.
        """
        value = kwargs.pop('value', _MISSING)
        ctor_kwargs = self._get_constructor_kwargs()
        ctor_kwargs.update(kwargs)
        if value is None:
            raise TypeError('QuantTensor value must be a Tensor, not None.')
        new_value = self._value if value is _MISSING else value
        return self._reconstruct(new_value, ctor_kwargs)

    def detach_(self):
        super().detach_()
        for attribute in self._constructor_metadata.values():
            val = getattr(self, attribute)
            if isinstance(val, Tensor):
                val.detach_()
        return self

    def detach(self):
        return self._apply_and_reconstruct(Tensor.detach)

    def contiguous(self):
        return self._apply_and_reconstruct(Tensor.contiguous)

    def _slice_constructor_kwargs(self, index):
        """Slice tensor metadata that is aligned with the leading value dimension."""
        original_shape = self.value.shape
        ctor_kwargs = self._get_constructor_kwargs()
        for name, metadata in tuple(ctor_kwargs.items()):
            if (isinstance(metadata, Tensor) and metadata.dim() > 0 and
                    metadata.shape[0] == original_shape[0]):
                ctor_kwargs[name] = metadata[index]
        return ctor_kwargs

    def __getitem__(self, index):
        """Index the leading dimension while preserving aligned quantization metadata."""
        if isinstance(index, Tensor):
            if index.dim() != 0 or index.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError('QuantTensor indices must be integer scalars.')
            index = int(index.item())
        if not isinstance(index, (int, slice)):
            raise TypeError('QuantTensor indexing supports an integer or slice.')

        return self._reconstruct(self._value[index], self._slice_constructor_kwargs(index))

    @property
    def shape(self):
        return self.value.shape

    def dim(self):
        return self.value.dim()

    def add(self, other):
        return self + other

    def to(self, *args, **kwargs):
        new_value = Tensor.to(self._value, *args, **kwargs)

        def transform_metadata(_, metadata):
            if metadata.dtype == torch.bool:
                return metadata.to(device=new_value.device)
            return metadata.to(*args, **kwargs)

        return self._reconstruct(new_value, metadata_transform=transform_metadata)

    def cuda(self, *args, **kwargs):
        return self._apply_and_reconstruct(Tensor.cuda, *args, **kwargs)

    def cpu(self, *args, **kwargs):
        return self._apply_and_reconstruct(Tensor.cpu, *args, **kwargs)

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


class GroupwiseQuantTensorMixin:
    """Implement behavior shared by QuantTensors with compressed groupwise storage."""

    def _set_groupwise_metadata(
            self, scale, zero_point, group_size, group_dim, signed, training, dequant_shape):
        """Store groupwise metadata shared by integer and floating-point formats."""
        self._scale = scale
        self._zero_point = zero_point
        self._group_size = group_size
        self._group_dim = group_dim
        self._signed = signed
        self._training = training
        self._dequant_shape = dequant_shape

    @property
    def group_size(self):
        return self._group_size

    @property
    def group_dim(self):
        return self._group_dim

    @property
    def dequant_shape(self):
        return self._dequant_shape

    @property
    def signed(self):
        return self._signed.item()

    @property
    def training(self):
        return self._training.item()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        from brevitas.quant_tensor import _unpack_quant_tensor

        from .torch_handler import QUANT_TENSOR_FN_HANDLER
        if func in QUANT_TENSOR_FN_HANDLER:
            return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)
        args = _unpack_quant_tensor(args)
        kwargs = _unpack_quant_tensor(kwargs)
        return func(*args, **kwargs)

    def expand(self, expand_metadata=True):
        """Expand grouped storage and optionally its quantization metadata."""
        from brevitas.utils.quant_utils import groupwise_dequant_expand

        return groupwise_dequant_expand(
            self._value,
            self._scale,
            self._zero_point,
            self.group_dim,
            self.dequant_shape,
            expand_metadata=expand_metadata)

    @staticmethod
    def from_expanded(value, group_size, group_dim, compress=False):
        """Convert an expanded value or metadata tensor to grouped storage."""
        group_dim = group_dim if group_dim >= 0 else group_dim - 1
        size = list(value.shape)
        assert size[group_dim] % group_size == 0, 'Input channel is not divisible by group size'
        if compress:
            size[group_dim] = 1
        else:
            size[group_dim] = size[group_dim] // group_size
        size.insert(group_dim + 1, group_size)
        return value.view(size)

    @property
    def value(self):
        value, _, _ = self.expand(expand_metadata=False)
        return value

    @property
    def scale(self):
        _, scale, _ = self.expand()
        return scale

    @property
    def zero_point(self):
        _, _, zero_point = self.expand()
        return zero_point

    @property
    def device(self):
        value_device = self._value.device
        if not self._metadata_on_device(value_device):
            raise RuntimeError("Value and metadata are on different devices")
        return value_device

    @classmethod
    def check_input_type(cls, tensor):
        if not isinstance(tensor, cls):
            raise RuntimeError(f"Tensor is not a {cls.__name__}")

    def view(self, *args, **kwargs):
        return self.value.view(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self.value.reshape(*args, **kwargs)

    def flatten(self, *args, **kwargs):
        return self.value.flatten(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self.value.transpose(*args, **kwargs)

    def permute(self, *args, **kwargs):
        return self.value.permute(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, QuantTensor):
            return self.value + other.value
        return self.value + other

    def __mul__(self, other):
        if isinstance(other, QuantTensor):
            return self.value * other.value
        return self.value * other

    def __truediv__(self, other):
        if isinstance(other, QuantTensor):
            return self.value / other.value
        return self.value / other

    def _slice_constructor_kwargs(self, index):
        """Update group geometry after indexing the dequantized leading dimension."""
        ctor_kwargs = super()._slice_constructor_kwargs(index)
        if isinstance(index, int):
            group_dim = self.group_dim
            original_shape = self.value.shape
            normalized_group_dim = group_dim if group_dim >= 0 else group_dim + len(original_shape)
            if normalized_group_dim == 0:
                raise RuntimeError('Cannot remove the grouped dimension through indexing.')
            ctor_kwargs['group_dim'] = group_dim - 1 if group_dim > 0 else group_dim
            if self.dequant_shape is not None:
                ctor_kwargs['dequant_shape'] = tuple(self.dequant_shape[1:])
        elif self.dequant_shape is not None:
            ctor_kwargs['dequant_shape'] = tuple(self.value[index].shape)
        return ctor_kwargs


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
                if self.bit_width <= 8. and self._signed.item():
                    return int_value.to(torch.int8)
                elif self.bit_width <= 8. and not self._signed.item():
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
        if not self._metadata_on_device(value_device):
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
