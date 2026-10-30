# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

from .float_quant_tensor import FloatQuantTensor
from .groupwise_float_quant_tensor import GroupwiseFloatQuantTensor
from .groupwise_int_quant_tensor import GroupwiseIntQuantTensor
from .int_quant_tensor import IntQuantTensor


def _immutable(value):
    """Convert container metadata to a form suitable for a flatten context."""
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, dict):
        return tuple((_immutable(key), _immutable(item)) for key, item in value.items())
    if isinstance(value, set):
        return frozenset(_immutable(item) for item in value)
    return value


def _aten_overload(name, overload='default'):
    """Look up an ATen default overload without assuming that it exists."""
    packet = getattr(torch.ops.aten, name, None)
    return getattr(packet, overload, None)


_PRESERVE_WRAPPER_OPS = frozenset(
    op for op in (_aten_overload('alias'), _aten_overload('clone'), _aten_overload('detach'))
    if op is not None)
_CONTIGUOUS_OP = _aten_overload('contiguous')
_TO_PRESERVE_WRAPPER_OPS = frozenset(
    op for op in (
        _aten_overload('_to_copy'),
        _aten_overload('to', 'dtype_layout'),
    ) if op is not None)


class TraceableQuantTensorMixin:
    """Non-storage Tensor-subclass implementation for traceable QuantTensors."""

    # Bypass the QuantTensor __torch_function__ handlers. ATen operations are handled by
    # __torch_dispatch__, where unsupported operations deliberately lose quantization metadata.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def _logical_size(value, args, kwargs):
        outer_size = kwargs.get('dequant_shape')
        cls = kwargs.pop('_traceable_cls', None)
        if outer_size is None and cls is not None and '_dequant_shape' in cls._quant_tensor_metadata:
            dequant_shape_index = cls._quant_tensor_metadata.index('_dequant_shape')
            if dequant_shape_index < len(args):
                outer_size = args[dequant_shape_index]
        return tuple(value.size()) if outer_size is None else tuple(outer_size)

    @staticmethod
    def _contiguous_stride(size):
        stride = 1
        strides = []
        for dim_size in reversed(size):
            strides.append(stride)
            stride *= dim_size
        return tuple(reversed(strides))

    @torch._dynamo.disable
    def __new__(cls, value, *args, _outer_size=None, _outer_stride=None, **kwargs):
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float)
        if isinstance(value, TraceableQuantTensorMixin):
            value = value._value
        logical_size_kwargs = dict(kwargs)
        logical_size_kwargs['_traceable_cls'] = cls
        outer_size = (
            TraceableQuantTensorMixin._logical_size(value, args, logical_size_kwargs)
            if _outer_size is None else tuple(_outer_size))
        if _outer_stride is None and kwargs.get('dequant_shape') is None and (
                '_dequant_shape' not in cls._quant_tensor_metadata or
                cls._quant_tensor_metadata.index('_dequant_shape') >= len(args)):
            outer_stride = tuple(value.stride())
            outer_storage_offset = value.storage_offset()
        else:
            outer_stride = (
                TraceableQuantTensorMixin._contiguous_stride(outer_size)
                if _outer_stride is None else tuple(_outer_stride))
            outer_storage_offset = 0
        return torch.Tensor._make_wrapper_subclass(
            cls,
            outer_size,
            strides=outer_stride,
            storage_offset=outer_storage_offset,
            dtype=value.dtype,
            layout=value.layout,
            device=value.device,
            requires_grad=value.requires_grad)

    @torch._dynamo.disable
    def __init__(self, value, *args, _outer_size=None, _outer_stride=None, **kwargs):
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float)
        if isinstance(value, TraceableQuantTensorMixin):
            value = value._value
        self._inner_value = value
        logical_size_kwargs = dict(kwargs)
        logical_size_kwargs['_traceable_cls'] = type(self)
        self._outer_size = (
            self._logical_size(value, args, logical_size_kwargs)
            if _outer_size is None else tuple(_outer_size))
        if _outer_stride is not None:
            self._outer_stride = tuple(_outer_stride)
        elif '_dequant_shape' in type(self)._quant_tensor_metadata:
            self._outer_stride = self._contiguous_stride(self._outer_size)
        else:
            self._outer_stride = tuple(value.stride())
        super().__init__(value, *args, **kwargs)

    @property
    def _value(self):
        """Return the explicitly stored raw value Tensor."""
        return self._inner_value

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

    def __tensor_flatten__(self):
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
    def __tensor_unflatten__(
            cls, inner_tensors, non_tensor_metadata, outer_size=None, outer_stride=None):
        metadata = dict(non_tensor_metadata)
        metadata.update({
            attribute: inner_tensors[attribute]
            for attribute in cls._quant_tensor_metadata
            if attribute in inner_tensors})
        kwargs = {attribute[1:]: metadata[attribute] for attribute in cls._quant_tensor_metadata}
        return cls(
            inner_tensors['_inner_value'],
            _outer_size=outer_size,
            _outer_stride=outer_stride,
            **kwargs)

    @staticmethod
    def _reconstruct_from_transformed_tensors(quant_tensor, transform):
        tensor_attributes, non_tensor_metadata = quant_tensor.__tensor_flatten__()
        inner_tensors = {
            attribute: transform(attribute, getattr(quant_tensor, attribute))
            for attribute in tensor_attributes}
        return type(quant_tensor).__tensor_unflatten__(
            inner_tensors, non_tensor_metadata, None, None)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is _CONTIGUOUS_OP:
            quant_tensor = args[0]
            output = TraceableQuantTensorMixin._reconstruct_from_transformed_tensors(
                quant_tensor,
                lambda attribute,
                value: (
                    func(value, *args[1:], **kwargs)
                    if attribute == '_inner_value' else Tensor.contiguous(value)))
            return return_and_correct_aliasing(func, args, kwargs, output)

        if func in _TO_PRESERVE_WRAPPER_OPS:
            quant_tensor = args[0]

            def move(attribute, value):
                move_kwargs = dict(kwargs)
                move_kwargs.pop('memory_format', None)
                if attribute != '_inner_value' and value.dtype == torch.bool:
                    move_kwargs.pop('dtype', None)
                return func(value, *args[1:], **move_kwargs)

            output = TraceableQuantTensorMixin._reconstruct_from_transformed_tensors(
                quant_tensor, move)
            return return_and_correct_aliasing(func, args, kwargs, output)

        if func in _PRESERVE_WRAPPER_OPS:
            quant_tensor = args[0]
            output = TraceableQuantTensorMixin._reconstruct_from_transformed_tensors(
                quant_tensor, lambda _, value: func(value, *args[1:], **kwargs))
            return return_and_correct_aliasing(func, args, kwargs, output)

        def unwrap(value):
            if isinstance(value, TraceableQuantTensorMixin):
                return value.value
            return value

        return func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))


class TraceableIntQuantTensor(TraceableQuantTensorMixin, IntQuantTensor):
    pass


class TraceableFloatQuantTensor(TraceableQuantTensorMixin, FloatQuantTensor):
    pass


class TraceableGroupwiseIntQuantTensor(TraceableQuantTensorMixin, GroupwiseIntQuantTensor):
    pass


class TraceableGroupwiseFloatQuantTensor(TraceableQuantTensorMixin, GroupwiseFloatQuantTensor):
    pass


_TRACEABLE_QUANT_TENSOR_CLASSES = {
    IntQuantTensor: TraceableIntQuantTensor,
    FloatQuantTensor: TraceableFloatQuantTensor,
    GroupwiseIntQuantTensor: TraceableGroupwiseIntQuantTensor,
    GroupwiseFloatQuantTensor: TraceableGroupwiseFloatQuantTensor,}


def get_traceable_quant_tensor_class(base_cls):
    """Return the traceable wrapper corresponding to a concrete QuantTensor class."""
    if base_cls in _TRACEABLE_QUANT_TENSOR_CLASSES.values():
        return base_cls
    try:
        return _TRACEABLE_QUANT_TENSOR_CLASSES[base_cls]
    except KeyError as exc:
        raise ValueError(f"Unsupported QuantTensor class: {base_cls!r}") from exc


def get_base_quant_tensor_class(quant_tensor_cls):
    """Return the eager QuantTensor class corresponding to a traceable wrapper."""
    for base_cls, traceable_cls in _TRACEABLE_QUANT_TENSOR_CLASSES.items():
        if quant_tensor_cls is traceable_cls or quant_tensor_cls is base_cls:
            return base_cls
    raise ValueError(f"Unsupported QuantTensor class: {quant_tensor_cls!r}")


__all__ = [
    'TraceableIntQuantTensor',
    'TraceableFloatQuantTensor',
    'TraceableGroupwiseIntQuantTensor',
    'TraceableGroupwiseFloatQuantTensor',
    'get_traceable_quant_tensor_class',
    'get_base_quant_tensor_class',]
