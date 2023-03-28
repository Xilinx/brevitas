# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.function_wrapper import *
from brevitas.core.quant import RescalingIntQuant
from brevitas.inject.enum import FloatToIntImplType


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
        return DPURoundSte
    else:
        return None
