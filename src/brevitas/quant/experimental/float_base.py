# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper import FloatClamp
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.float import FloatQuant
from brevitas.core.scaling.float_scaling import FloatScaling
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from brevitas.quant.solver.common import SolveTensorQuantFloatToIntImplFromEnum
from brevitas.utils.float_quant_utils import get_max_value


class FloatWeightBase(SolveTensorQuantFloatToIntImplFromEnum):
    proxy_class = WeightQuantProxyFromInjector
    tensor_quant = FloatQuant
    signed = True
    float_to_int_impl_type = 'round'
    scaling_min_val = 1e-10


class FloatActBase(SolveTensorQuantFloatToIntImplFromEnum):
    proxy_class = ActQuantProxyFromInjector
    tensor_quant = FloatQuant
    signed = True
    float_to_int_impl_type = 'round'
    scaling_min_val = 1e-10


class ScaledFloatWeightBase(FloatWeightBase, WeightQuantSolver):
    scaling_stats_op = 'max'
    scaling_impl_type = 'stats'
    restrict_scaling_type = 'fp'
    float_scaling_impl = FloatScaling


class ScaledFloatActBase(FloatActBase, ActQuantSolver):
    scaling_stats_op = 'max'
    scaling_impl_type = 'parameter_from_stats'
    restrict_scaling_type = 'fp'
    collect_stats_steps = 300
    float_scaling_impl = FloatScaling


class ExponentBiasMixin(ExtendedInjector):

    @value
    def exponent_bias(exponent_bit_width):
        return 2 ** (exponent_bit_width - 1) - 1


class MaxFloatInfNaNMixin(ExtendedInjector):

    @value
    def max_value(
            exponent_bit_width, mantissa_bit_width, exponent_bias, nan_values, inf_values,
            saturating):
        return get_max_value(
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            nan_values,
            inf_values,
            saturating)


class Fp8e4m3Mixin(ExponentBiasMixin, MaxFloatInfNaNMixin):
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3
    float_clamp_impl = FloatClamp
    tensor_clamp_impl = TensorClamp
    nan_values = (('111',))
    inf_values = None
    saturating = True


class Fp8e5m2Mixin(ExponentBiasMixin, MaxFloatInfNaNMixin):
    bit_width = 8
    exponent_bit_width = 5
    mantissa_bit_width = 2
    float_clamp_impl = FloatClamp
    tensor_clamp_impl = TensorClamp
    nan_values = ('01', '11', '10')
    inf_values = (('00',))
    saturating = True
