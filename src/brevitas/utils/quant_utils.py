# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

import brevitas.compiler as brevitas_compiler
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
        if metadata_only:
            self.value = None
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor
            # torch.compile compatibility
            self.value = quant_tensor.value
        # torch.compile compatibility
        self.scale = quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


class _CachedIOFloat:

    def __init__(self, quant_tensor: FloatQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.value = None
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor
            # torch.compile compatibility
            self.value = quant_tensor.value
        # torch.compile compatibility
        self.scale = quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def exponent_bit_width(self):
        return self.quant_tensor.exponent_bit_width

    @property
    def mantissa_bit_width(self):
        return self.quant_tensor.mantissa_bit_width

    @property
    def exponent_bias(self):
        return self.quant_tensor.exponent_bias

    @property
    def saturating(self):
        return self.quant_tensor.saturating

    @property
    def inf_values(self):
        return self.quant_tensor.inf_values

    @property
    def nan_values(self):
        return self.quant_tensor.nan_values

    @property
    def signed(self):
        return self.quant_tensor.signed


class _CachedIOGroupwiseFloat:

    def __init__(self, quant_tensor: GroupwiseFloatQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.value = None
            self.quant_tensor = quant_tensor.set(value_=None)
        else:
            self.quant_tensor = quant_tensor
            # torch.compile compatibility
            self.value = quant_tensor.value
        # torch.compile compatibility
        self.scale_ = quant_tensor.scale_
        self.zero_point_ = quant_tensor.zero_point_

    @property
    def exponent_bit_width(self):
        return self.quant_tensor.exponent_bit_width

    @property
    def mantissa_bit_width(self):
        return self.quant_tensor.mantissa_bit_width

    @property
    def exponent_bias(self):
        return self.quant_tensor.exponent_bias

    @property
    def saturating(self):
        return self.quant_tensor.saturating

    @property
    def inf_values(self):
        return self.quant_tensor.inf_values

    @property
    def nan_values(self):
        return self.quant_tensor.nan_values

    @property
    def signed(self):
        return self.quant_tensor.signed

    @property
    def group_size(self):
        return self.quant_tensor.group_size

    @property
    def group_dim(self):
        return self.quant_tensor.group_dim


class _CachedIOGroupwiseInt:

    def __init__(self, quant_tensor: GroupwiseIntQuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.value = None
            self.quant_tensor = quant_tensor.set(value_=None)
        else:
            self.quant_tensor = quant_tensor
            # torch.compile compatibility
            self.value = quant_tensor.value
        # torch.compile compatibility
        self.scale_ = quant_tensor.scale_
        self.zero_point_ = quant_tensor.zero_point_

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed

    @property
    def group_size(self):
        return self.quant_tensor.group_size

    @property
    def group_dim(self):
        return self.quant_tensor.group_dim


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


# For old versions of pytorch (2.3.1), this is needed otherwise compile skips this function
@brevitas_compiler.disable
def groupwise_dequant_expand(value_, scale_, zero_point_, group_dim, dequant_shape):
    curr_shape = value_.shape
    start_dim = group_dim if group_dim >= 0 else group_dim - 1
    new_value = value_.flatten(start_dim, start_dim + 1)
    if scale_.shape != ():
        new_scale = scale_.expand(curr_shape).flatten(start_dim, start_dim + 1)
    else:
        new_scale = scale_
    if zero_point_.shape != ():
        new_zp = zero_point_.expand(curr_shape).flatten(start_dim, start_dim + 1)
    else:
        new_zp = zero_point_

    # If we padded during quantization, we unpad here:
    # First, we compute how much we padded along the group_dim shape
    # Then, we unbind the tensor along the group_dim shape, and drop the padded columns
    # Finally, we stack the remaining tensors
    unpadding_shape = dequant_shape[group_dim]
    residual = new_value.shape[group_dim] - unpadding_shape

    if residual > 0:
        new_value = torch.stack(
            torch.unbind(new_value, dim=group_dim)[:unpadding_shape], dim=group_dim)
        new_scale = torch.stack(
            torch.unbind(new_scale, dim=group_dim)[:unpadding_shape], dim=group_dim)
        if zero_point_.shape != ():
            new_zp = torch.stack(
                torch.unbind(new_zp, dim=group_dim)[:unpadding_shape], dim=group_dim)

    return new_value, new_scale, new_zp
