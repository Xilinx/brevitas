from typing import List, NamedTuple, Optional

from torch import Tensor


# Base class for all QuantTensor.
# Only assumptions made by these methods are:
# - `self` is a NamedTuple with a `_fields` attribute
# - `self` has a `value` attribute
class QuantTensor:

    def detach_(self):
        for field in self._fields:
            getattr(self, field).detach_()

    def detach(self):
        qt_type = type(self)
        values = []
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, Tensor):
                value = value.detach()
            values.append(value)
        return qt_type(*values)

    def contiguous(self):
        qt_type = type(self)
        values = []
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, Tensor):
                value = value.contiguous()
            values.append(value)
        return qt_type(*values)

    def set(self, **kwargs):
        return self._replace(**kwargs)

    @property
    def shape(self):
        return self.value.shape

    def dim(self):
        return self.value.dim()

    def add(self, other):
        return self + other

    def to(self, *args, **kwargs):
        qt_type = type(self)
        values = []
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, Tensor):
                value = value.to(*args, **kwargs)
            values.append(value)
        return qt_type(*values)

    def cuda(self, *args, **kwargs):
        qt_type = type(self)
        values = []
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, Tensor):
                value = value.cuda(*args, **kwargs)
            values.append(value)
        return qt_type(*values)

    def cpu(self, *args, **kwargs):
        qt_type = type(self)
        values = []
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, Tensor):
                value = value.cpu(*args, **kwargs)
            values.append(value)
        return qt_type(*values)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __pos__(self):
        return self

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)


class IntQuantTensorBase(NamedTuple):
    value: Tensor
    scale: Tensor
    zero_point: Tensor
    bit_width: Tensor
    signed_t: Tensor
    training_t: Tensor


class FloatQuantTensorBase(NamedTuple):
    value: Tensor
    scale: Tensor
    zero_point: Tensor
    exponent_bit_width: Tensor
    mantissa_bit_width: Tensor
    exponent_bias: Tensor
    saturating_t: Tensor
    inf_values: List[str]
    nan_values: List[str]
    signed_t: Tensor
    training_t: Tensor


class GroupwiseFloatQuantTensorBase(NamedTuple):
    value_: Tensor
    scale_: Tensor
    zero_point_: Tensor
    group_size: Tensor
    group_dim: Tensor
    exponent_bit_width: Tensor
    mantissa_bit_width: Tensor
    exponent_bias: Tensor
    saturating_t: Tensor
    inf_values: List[str]
    nan_values: List[str]
    signed_t: Tensor
    training_t: Tensor


class GroupwisIntQuantTensorBase(NamedTuple):
    value_: Tensor
    scale_: Tensor
    zero_point_: Tensor
    group_size: Tensor
    group_dim: Tensor
    bit_width: Tensor
    signed_t: Tensor
    training_t: Tensor


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
