# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.function_wrapper import *
from brevitas.core.quant import RescalingIntQuant
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.function.shape import over_output_channels
from brevitas.inject.enum import FloatToIntImplType
from brevitas.quant_tensor import QuantTensor


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


def has_learned_weight_bit_width(module):
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector

    if isinstance(module, WeightQuantProxyFromInjector) \
            and isinstance(module.tensor_quant, RescalingIntQuant) \
            and isinstance(module.tensor_quant.msb_clamp_bit_width_impl,
                           BitWidthParameter):
        return True
    else:
        return False


def has_learned_activation_bit_width(module):
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
    from brevitas.proxy.runtime_quant import FusedActivationQuantProxy

    if isinstance(module, ActQuantProxyFromInjector) \
            and isinstance(module.fused_activation_quant_proxy, FusedActivationQuantProxy) \
            and isinstance(module.fused_activation_quant_proxy.tensor_quant, RescalingIntQuant) \
            and isinstance(module.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl,
                           BitWidthParameter):
        return True
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


def _calculate_acc_range(module, input_range):
    quant_weight = module.quant_weight()
    quant_weight: Tensor = quant_weight.int().float()
    shape = over_output_channels(quant_weight)
    quant_weight = quant_weight.reshape(shape)

    max_vectors = torch.where(quant_weight > 0, input_range[1], input_range[0])
    min_vectors = torch.where(quant_weight > 0, input_range[0], input_range[1])
    max_values_per_accumulator: Tensor = (quant_weight * max_vectors).sum(axis=1)
    min_values_per_accumulator: Tensor = (quant_weight * min_vectors).sum(axis=1)

    max_value = max_values_per_accumulator.max()
    min_value = min_values_per_accumulator.min()
    return (min_value, max_value)


def calculate_accumulator_bit_width(module) -> Tensor:
    input_bit_width = module.quant_input_bit_width()
    input_is_signed = module.is_quant_input_signed
    input_is_narrow = module.is_quant_input_narrow_range
    if input_bit_width is not None:
        input_min = min_int(input_is_signed, input_is_narrow, input_bit_width)
        input_max = max_int(input_is_signed, input_is_narrow, input_bit_width)
        (acc_min, acc_max) = _calculate_acc_range(module, [input_min, input_max])
        _acc_max = max(-acc_min, 1 + acc_max)
        acc_bit_width = torch.log2(_acc_max) + 1
        acc_bit_width = torch.ceil(acc_bit_width)
        return acc_bit_width
    return torch.tensor([32.0])
