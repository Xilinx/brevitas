# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

import brevitas.compiler as brevitas_compiler
from brevitas.function.ops_ste import round_ste
from brevitas.utils.torch_utils import float_internal_scale

TOLERANCE = {torch.float64: 1e-1, torch.float32: 2e-1, torch.float16: 0.5, torch.bfloat16: 0.5}
_MISSING = object()


def _immutable(value):
    """Convert container metadata to a hashable form for a flatten context."""
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, dict):
        return tuple((_immutable(key), _immutable(item)) for key, item in value.items())
    if isinstance(value, set):
        return frozenset(_immutable(item) for item in value)
    return value


def _contiguous_stride(size):
    stride = 1
    strides = []
    for dim_size in reversed(size):
        strides.append(stride)
        stride *= dim_size
    return tuple(reversed(strides))


def _ungrouped_shape(shape, group_dim):
    """Return the shape of grouped storage once its group dimensions are merged."""
    start_dim = (group_dim if group_dim >= 0 else group_dim - 1) % len(shape)
    return (*shape[:start_dim], shape[start_dim] * shape[start_dim + 1], *shape[start_dim + 2:])


# ATen operations that return a QuantTensor of the same type, with every inner tensor transformed.
_PRESERVE_WRAPPER_OPS = frozenset((
    torch.ops.aten.alias.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.detach.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype_layout,
))


# Base class for all QuantTensor types.
# Wrapper Tensor subclass without storage, whose shape is the logical (dequantized) one.
# The raw data (dequantized for regular QuantTensors and grouped for groupwise QuantTensors) and the
# tensor-valued quantization metadata (scale, zero_point, bit_width, etc.) are kept as inner
# tensors, which __tensor_flatten__ and __tensor_unflatten__ expose to torch.compile.
# Other metadata (signed, training, group settings, etc.) is stored as plain Python values.
class QuantTensor(Tensor):

    # Private metadata attributes used to reconstruct concrete QuantTensor types.
    # Their order matches the constructor arguments that follow ``value``.
    _quant_tensor_metadata = ()

    @staticmethod
    @brevitas_compiler.disable
    def __new__(cls, value, *args, _outer_size=None, _outer_stride=None, **kwargs):
        # _outer_size and _outer_stride are only passed by __tensor_unflatten__.
        if cls is QuantTensor:
            raise TypeError('QuantTensor is a base class and cannot be instantiated directly.')
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float)
        if isinstance(value, QuantTensor):
            value = value._value
        size, stride, storage_offset = cls._logical_geometry(value, _outer_size, args, kwargs)
        if _outer_stride is not None:
            stride, storage_offset = tuple(_outer_stride), 0
        quant_tensor = Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            storage_offset=storage_offset,
            dtype=value.dtype,
            layout=value.layout,
            device=value.device,
            requires_grad=value.requires_grad)
        quant_tensor._inner_value = value
        quant_tensor._outer_size = size
        quant_tensor._outer_stride = stride
        return quant_tensor

    def __init__(self, value, *args, **kwargs):
        # Subclasses should NOT call super().__init__() with args;
        # metadata is set by subclass __init__ methods.
        pass

    @classmethod
    def _logical_geometry(cls, value, size, args, kwargs):
        """
        Return the size, stride and storage offset advertised by the wrapper around ``value``.

        ``size`` is the logical size requested when a QuantTensor is rebuilt, None otherwise.
        """
        # The raw value of regular QuantTensors has the logical shape and layout.
        size = tuple(value.size()) if size is None else tuple(size)
        return size, tuple(value.stride()), value.storage_offset()

    @property
    def _value(self):
        """Return the raw Tensor representation stored by this Tensor subclass."""
        return self._inner_value

    @property
    def value(self):
        return self._value

    @property
    def signed(self):
        return self._signed

    @property
    def training(self):
        return self._training

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

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in _PRESERVE_WRAPPER_OPS:
            # Memory formats only apply to the raw value, not to its metadata.
            metadata_kwargs = dict(kwargs)
            metadata_kwargs.pop('memory_format', None)

            def transform(attribute, value):
                if attribute == '_inner_value':
                    return func(value, *args[1:], **kwargs)
                return func(value, *args[1:], **metadata_kwargs)

            output = cls._transform_inner_tensors(args[0], transform)
            return return_and_correct_aliasing(func, args, kwargs, output)

        # Any other operation that reaches the dispatcher acts on the dequantized value.
        return func(*_unpack_quant_tensor(args), **_unpack_quant_tensor(kwargs))

    def __tensor_flatten__(self):
        """Split the metadata into inner tensors and a hashable context."""
        tensor_attributes = ['_inner_value']
        non_tensor_metadata = []
        for attribute in self._quant_tensor_metadata:
            value = getattr(self, attribute)
            if isinstance(value, Tensor):
                tensor_attributes.append(attribute)
            else:
                non_tensor_metadata.append((attribute, _immutable(value)))
        return tensor_attributes, tuple(non_tensor_metadata)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, non_tensor_metadata, outer_size, outer_stride):
        # torch always passes the logical size and stride. Without them, the logical size of a
        # groupwise QuantTensor is inferred as if its grouped dimension was not padded.
        metadata = dict(non_tensor_metadata)
        metadata.update({
            attribute: inner_tensors[attribute]
            for attribute in cls._quant_tensor_metadata
            if attribute in inner_tensors})
        kwargs = {attribute[1:]: metadata[attribute] for attribute in cls._quant_tensor_metadata}
        value = inner_tensors['_inner_value']
        # The concrete __init__ does not accept the outer geometry, so __new__ and __init__ are
        # called separately.
        quant_tensor = cls.__new__(
            cls, value, _outer_size=outer_size, _outer_stride=outer_stride, **kwargs)
        cls.__init__(quant_tensor, value, **kwargs)
        return quant_tensor

    @staticmethod
    def _transform_inner_tensors(quant_tensor, transform):
        """Rebuild ``quant_tensor`` with every inner tensor replaced by ``transform(name, tensor)``.

        The logical shape is preserved, while the layout follows the transformed raw value.
        """
        tensor_attributes, non_tensor_metadata = quant_tensor.__tensor_flatten__()
        inner_tensors = {
            attribute: transform(attribute, getattr(quant_tensor, attribute))
            for attribute in tensor_attributes}
        return type(quant_tensor).__tensor_unflatten__(
            inner_tensors, non_tensor_metadata, quant_tensor.shape, None)

    @staticmethod
    def _as_tensor(value, dtype, device):
        """Convert metadata literals without moving tensor-valued metadata."""
        return value if isinstance(value, Tensor) else torch.tensor(
            value, dtype=dtype, device=device)

    @staticmethod
    def _as_bool(value):
        """Convert boolean metadata, given as a bool or a single-element tensor, to a bool."""
        if value is None:
            raise TypeError('Boolean QuantTensor metadata cannot be None.')
        return bool(value)

    def _get_metadata_kwargs(self):
        """Return constructor metadata from its private backing attributes."""
        # Private metadata attributes map to constructor keyword names without the prefix.
        return {attribute[1:]: getattr(self, attribute) for attribute in self._quant_tensor_metadata}

    def _reconstruct(self, value, metadata_kwargs=None):
        """
        Rebuild this type from constructor-form value and metadata.

        ``value`` must use the representation expected by the concrete
        constructor: dequantized for regular QuantTensors and grouped for
        groupwise QuantTensors.
        """
        if metadata_kwargs is None:
            metadata_kwargs = self._get_metadata_kwargs()
        else:
            metadata_kwargs = dict(metadata_kwargs)
        return type(self)(value, **metadata_kwargs)

    def _apply_and_reconstruct(self, tensor_op, *args, **kwargs):
        """Apply a Tensor operation to the constructor value and tensor metadata."""
        metadata_kwargs = self._get_metadata_kwargs()
        for parameter, metadata in metadata_kwargs.items():
            if isinstance(metadata, Tensor):
                metadata_kwargs[parameter] = tensor_op(metadata, *args, **kwargs)
        return self._reconstruct(tensor_op(self._value, *args, **kwargs), metadata_kwargs)

    def _metadata_on_device(self, device):
        """Return whether every tensor-backed metadata field is on ``device``."""
        return all(
            device == getattr(self, attribute).device
            for attribute in self._quant_tensor_metadata
            if isinstance(getattr(self, attribute), Tensor))

    @property
    def device(self):
        value_device = self._value.device
        if not self._metadata_on_device(value_device):
            raise RuntimeError("Value and metadata are on different devices")
        return value_device

    def set(self, **kwargs):
        """
        Create a new QuantTensor with some fields replaced.

        ``value`` replaces the raw constructor value.
        """
        value = kwargs.pop('value', _MISSING)
        metadata_kwargs = self._get_metadata_kwargs()
        metadata_kwargs.update(kwargs)
        if value is None:
            raise TypeError('QuantTensor value must be a Tensor, not None.')
        new_value = self._value if value is _MISSING else value
        return self._reconstruct(new_value, metadata_kwargs)

    def detach_(self):
        super().detach_()
        for attribute in self._quant_tensor_metadata:
            val = getattr(self, attribute)
            if isinstance(val, Tensor):
                val.detach_()
        return self

    def detach(self):
        return self._apply_and_reconstruct(Tensor.detach)

    def contiguous(self, memory_format=torch.contiguous_format):
        """Return a contiguous QuantTensor without applying value layouts to metadata."""
        metadata_kwargs = self._get_metadata_kwargs()
        for parameter, metadata in metadata_kwargs.items():
            if isinstance(metadata, Tensor):
                metadata_kwargs[parameter] = Tensor.contiguous(metadata)
        return self._reconstruct(
            Tensor.contiguous(self._value, memory_format=memory_format), metadata_kwargs)

    @property
    def shape(self):
        return torch.Size(self._outer_size)

    def size(self, dim=None):
        size = torch.Size(self._outer_size)
        return size if dim is None else size[dim]

    def dim(self):
        return len(self._outer_size)

    def stride(self, dim=None):
        stride = self._outer_stride
        return stride if dim is None else stride[dim]

    def add(self, other, *, alpha=1):
        if alpha == 1:
            return self + other
        other = other.value if isinstance(other, QuantTensor) else other
        return torch.add(self.value, other, alpha=alpha)

    def to(self, *args, **kwargs):
        new_value = Tensor.to(self._value, *args, **kwargs)
        metadata_kwargs = self._get_metadata_kwargs()
        metadata_to_kwargs = dict(kwargs)
        metadata_to_kwargs.pop('memory_format', None)
        for parameter, metadata in metadata_kwargs.items():
            if isinstance(metadata, Tensor):
                metadata_kwargs[parameter] = metadata.to(*args, **metadata_to_kwargs)
        return self._reconstruct(new_value, metadata_kwargs)

    def cuda(self, *args, **kwargs):
        return self._apply_and_reconstruct(Tensor.cuda, *args, **kwargs)

    def cpu(self, *args, **kwargs):
        return self._apply_and_reconstruct(Tensor.cpu, *args, **kwargs)

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

    @staticmethod
    def is_zero_zero_point(tensor):
        return (tensor.zero_point == 0.).all()


class GroupwiseQuantTensorMixin:
    """Implement behavior shared by QuantTensors with compressed groupwise storage."""

    @classmethod
    def _logical_geometry(cls, value, size, args, kwargs):
        # The raw value is grouped, while the wrapper advertises the dequantized tensor.
        if size is None:
            size = cls._dequant_shape_argument(value, args, kwargs)
        size = tuple(size)
        return size, _contiguous_stride(size), 0

    @classmethod
    def _dequant_shape_argument(cls, value, args, kwargs):
        """Return the dequantized shape passed to the constructor, or infer it from ``value``."""

        def argument(name, position):
            if name in kwargs:
                return kwargs[name]
            return args[position] if position < len(args) else None

        # dequant_shape is the constructor argument that follows the quantization metadata.
        dequant_shape = argument('dequant_shape', len(cls._quant_tensor_metadata))
        if dequant_shape is not None:
            return dequant_shape
        group_dim = argument('group_dim', cls._quant_tensor_metadata.index('_group_dim'))
        if group_dim is None:
            # The missing argument is reported by __init__.
            return value.shape
        # Without a dequantized shape, the grouped dimension is assumed not to be padded.
        return _ungrouped_shape(value.shape, group_dim)

    def _get_metadata_kwargs(self):
        # The dequantized shape is not metadata, it is the logical shape of the wrapper.
        metadata_kwargs = super()._get_metadata_kwargs()
        metadata_kwargs['dequant_shape'] = self.shape
        return metadata_kwargs

    def _set_groupwise_metadata(self, scale, zero_point, group_size, group_dim, signed, training):
        """Store groupwise metadata shared by integer and floating-point formats."""
        self._scale = scale
        self._zero_point = zero_point
        self._group_size = group_size
        self._group_dim = group_dim
        self._signed = signed
        self._training = training

    @property
    def group_size(self):
        return self._group_size

    @property
    def group_dim(self):
        return self._group_dim

    @property
    def dequant_shape(self):
        return self.shape

    def _expand_groupwise(self, expand_metadata=True):
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
        value, _, _ = self._expand_groupwise(expand_metadata=False)
        return value

    @property
    def scale(self):
        _, scale, _ = self._expand_groupwise()
        return scale

    @property
    def zero_point(self):
        _, _, zero_point = self._expand_groupwise()
        return zero_point

    def expand(self, *sizes):
        return self.value.expand(*sizes)

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


class IntMixin:

    @property
    def bit_width(self):
        return self._bit_width

    @bit_width.setter
    def bit_width(self, value):
        self._bit_width = value

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
                if self.bit_width <= 8. and self.signed:
                    return int_value.to(torch.int8)
                elif self.bit_width <= 8. and not self.signed:
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
        return self._saturating

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
