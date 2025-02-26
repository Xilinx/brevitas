from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from brevitas.function.ops_ste import round_ste

IS_VALID_ATOL = 2e-1
BFLOAT16_IS_VALID_ATOL = 0.5

IS_VALID_ATOL = 2e-1
B_FLOAT16_IS_VALID_ATOL = 0.5


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

    @staticmethod
    def is_zero_zero_point(tensor):
        return (tensor.zero_point == 0.).all()


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
    dequant_shape: Optional[Tuple] = None


class GroupwisIntQuantTensorBase(NamedTuple):
    value_: Tensor
    scale_: Tensor
    zero_point_: Tensor
    group_size: Tensor
    group_dim: Tensor
    bit_width: Tensor
    signed_t: Tensor
    training_t: Tensor
    dequant_shape: Optional[Tuple] = None


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
