from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from brevitas.function.ops_ste import round_ste
from brevitas.utils.torch_utils import float_internal_scale

TOLERANCE = {torch.float32: 2e-1, torch.float16: 0.5, torch.bfloat16: 0.5}


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
