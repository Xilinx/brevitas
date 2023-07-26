# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper import RoundSte
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling.float_scaling import FloatScaling
from brevitas.inject import ExtendedInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver


class FloatWeightBase(ExtendedInjector):
    proxy_class = WeightQuantProxyFromInjector
    tensor_quant = FloatQuant
    signed = True
    float_to_int_impl = RoundSte


class FloatActBase(ExtendedInjector):
    proxy_class = ActQuantProxyFromInjector
    tensor_quant = FloatQuant
    signed = True
    float_to_int_impl = RoundSte


class ScaledFloatWeightBase(FloatWeightBase, WeightQuantSolver):
    scaling_stats_op = 'max'
    scaling_impl_type = 'stats'
    restrict_scaling_type = 'fp'
    float_scaling_impl = FloatScaling


class ScaledFloatActBase(FloatActBase, ActQuantSolver):
    scaling_stats_op = 'percentile'
    scaling_impl_type = 'parameter_from_stats'
    restrict_scaling_type = 'fp'
    high_percentile_q = 99.999
    collect_stats_steps = 300
    float_scaling_impl = FloatScaling


class Fp8e4m3Mixin(ExtendedInjector):
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3
    exponent_bias = 7


class Fp8e5m2Mixin(ExtendedInjector):
    bit_width = 8
    exponent_bit_width = 5
    mantissa_bit_width = 2
    exponent_bias = 15
