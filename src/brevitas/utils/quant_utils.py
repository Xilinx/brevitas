# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.function_wrapper import *
from brevitas.core.quant import RescalingIntQuant
from brevitas.inject.enum import FloatToIntImplType
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.quant_tensor import IntQuantTensor


class _CachedIO:

    def __init__(self, quant_tensor: IntQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        self.value = None if metadata_only else quant_tensor.value
        self._quant_tensor = None if metadata_only else quant_tensor
        self.scale = quant_tensor.scale
        self.zero_point = quant_tensor.zero_point
        self.bit_width = quant_tensor.bit_width
        self._signed = quant_tensor._signed
        self._training = quant_tensor._training

    @property
    def signed(self):
        return self._signed.item()

    @property
    def quant_tensor(self):
        if self._quant_tensor is not None:
            return self._quant_tensor
        value = self.value if self.value is not None else self.scale.new_empty(0)
        self._quant_tensor = IntQuantTensor(
            value, self.scale, self.zero_point, self.bit_width, self._signed, self._training)
        return self._quant_tensor


class _CachedIOFloat:

    def __init__(self, quant_tensor: FloatQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        self.value = None if metadata_only else quant_tensor.value
        self._quant_tensor = None if metadata_only else quant_tensor
        self.scale = quant_tensor.scale
        self.zero_point = quant_tensor.zero_point
        self.exponent_bit_width = quant_tensor.exponent_bit_width
        self.mantissa_bit_width = quant_tensor.mantissa_bit_width
        self.exponent_bias = quant_tensor.exponent_bias
        self.inf_values = quant_tensor.inf_values
        self.nan_values = quant_tensor.nan_values
        self._saturating = quant_tensor._saturating
        self._signed = quant_tensor._signed
        self._training = quant_tensor._training

    @property
    def saturating(self):
        return self._saturating.item()

    @property
    def signed(self):
        return self._signed.item()

    @property
    def quant_tensor(self):
        if self._quant_tensor is not None:
            return self._quant_tensor
        value = self.value if self.value is not None else self.scale.new_empty(0)
        self._quant_tensor = FloatQuantTensor(
            value,
            self.scale,
            self.zero_point,
            self.exponent_bit_width,
            self.mantissa_bit_width,
            self.exponent_bias,
            self._saturating,
            self.inf_values,
            self.nan_values,
            self._signed,
            self._training)
        return self._quant_tensor


class _CachedIOGroupwiseFloat:

    def __init__(self, quant_tensor: GroupwiseFloatQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        self.value = None if metadata_only else quant_tensor.value
        self._value = None if metadata_only else quant_tensor._value
        self._quant_tensor = None if metadata_only else quant_tensor
        self._scale = quant_tensor._scale
        self._zero_point = quant_tensor._zero_point
        self.exponent_bit_width = quant_tensor.exponent_bit_width
        self.mantissa_bit_width = quant_tensor.mantissa_bit_width
        self.exponent_bias = quant_tensor.exponent_bias
        self.inf_values = quant_tensor.inf_values
        self.nan_values = quant_tensor.nan_values
        self.group_size = quant_tensor.group_size
        self.group_dim = quant_tensor.group_dim
        self.dequant_shape = quant_tensor.dequant_shape
        self._saturating = quant_tensor._saturating
        self._signed = quant_tensor._signed
        self._training = quant_tensor._training

    @property
    def saturating(self):
        return self._saturating.item()

    @property
    def signed(self):
        return self._signed.item()

    @property
    def quant_tensor(self):
        if self._quant_tensor is not None:
            return self._quant_tensor
        value = self._value if self._value is not None else self._scale.new_empty(0)
        self._quant_tensor = GroupwiseFloatQuantTensor(
            value,
            self._scale,
            self._zero_point,
            self.group_size,
            self.group_dim,
            self.exponent_bit_width,
            self.mantissa_bit_width,
            self.exponent_bias,
            self._saturating,
            self.inf_values,
            self.nan_values,
            self._signed,
            self._training,
            self.dequant_shape)
        return self._quant_tensor


class _CachedIOGroupwiseInt:

    def __init__(self, quant_tensor: GroupwiseIntQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        self.value = None if metadata_only else quant_tensor.value
        self._value = None if metadata_only else quant_tensor._value
        self._quant_tensor = None if metadata_only else quant_tensor
        self._scale = quant_tensor._scale
        self._zero_point = quant_tensor._zero_point
        self.bit_width = quant_tensor.bit_width
        self.group_size = quant_tensor.group_size
        self.group_dim = quant_tensor.group_dim
        self.dequant_shape = quant_tensor.dequant_shape
        self._signed = quant_tensor._signed
        self._training = quant_tensor._training

    @property
    def signed(self):
        return self._signed.item()

    @property
    def quant_tensor(self):
        if self._quant_tensor is not None:
            return self._quant_tensor
        value = self._value if self._value is not None else self._scale.new_empty(0)
        self._quant_tensor = GroupwiseIntQuantTensor(
            value,
            self._scale,
            self._zero_point,
            self.group_size,
            self.group_dim,
            self.bit_width,
            self._signed,
            self._training,
            self.dequant_shape)
        return self._quant_tensor


def has_learned_weight_bit_width(module):
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase

    if isinstance(module, WeightQuantProxyFromInjectorBase):
        for m in module.modules():
            if isinstance(m, BitWidthParameter):
                return True
        return False
    else:
        return False


def has_learned_activation_bit_width(module):
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
    from brevitas.proxy.runtime_quant import FusedActivationQuantProxy

    if isinstance(module, ActQuantProxyFromInjectorBase):
        for m in module.modules():
            if isinstance(m, BitWidthParameter):
                return True
        return False
    else:
        return False


def float_to_int_impl_to_enum(module):
    if isinstance(module, RoundSte):
        return FloatToIntImplType.ROUND
    elif isinstance(module, RoundToZeroSte):
        return FloatToIntImplType.ROUND_TO_ZERO
    elif isinstance(module, FloorSte):
        return FloatToIntImplType.FLOOR
    elif isinstance(module, CeilSte):
        return FloatToIntImplType.CEIL
    elif isinstance(module, DPURoundSte):
        return FloatToIntImplType.DPU
    elif isinstance(module, LearnedRoundSte):
        return FloatToIntImplType.LEARNED_ROUND
    elif isinstance(module, StochasticRoundSte):
        if module.deterministic_inference:
            return FloatToIntImplType.ROUND
        else:
            return FloatToIntImplType.STOCHASTIC_ROUND
    else:
        return None


def groupwise_dequant_expand(
        value_, scale_, zero_point_, group_dim, dequant_shape, expand_metadata=True):
    curr_shape = value_.shape
    start_dim = group_dim if group_dim >= 0 else group_dim - 1
    new_value = value_.flatten(start_dim, start_dim + 1)

    unpadding_shape = dequant_shape[group_dim]
    new_value = torch.narrow(new_value, group_dim, 0, unpadding_shape)

    if not expand_metadata:
        return new_value, scale_, zero_point_

    if scale_.shape != ():
        new_scale = scale_.expand(curr_shape).flatten(start_dim, start_dim + 1)
    else:
        new_scale = scale_
    if zero_point_.shape != ():
        new_zp = zero_point_.expand(curr_shape).flatten(start_dim, start_dim + 1)
    else:
        new_zp = zero_point_

    if scale_.shape != ():
        new_scale = torch.narrow(new_scale, group_dim, 0, unpadding_shape)
    if zero_point_.shape != ():
        new_zp = torch.narrow(new_zp, group_dim, 0, unpadding_shape)

    return new_value, new_scale, new_zp
