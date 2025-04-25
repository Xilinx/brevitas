# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.quant import TruncIntQuant
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.core.scaling import TruncMsbScaling
from brevitas.core.scaling import TruncScalingWrapper
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import TruncScalingImplType
from brevitas.proxy import TruncQuantProxyFromInjector
from brevitas.quant.solver.act import *
from brevitas.quant.solver.common import *


class SolveTruncTensorQuantFromEnum(ExtendedInjector):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return TruncIntQuant
        elif quant_type == QuantType.TERNARY:
            raise RuntimeError(f'{quant_type} not supported for truncation.')
        elif quant_type == QuantType.BINARY:
            raise RuntimeError(f'{quant_type} not supported for truncation.')
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class SolveTruncScalingImplFromEnum(ExtendedInjector):

    @value
    def trunc_scaling_impl(trunc_scaling_impl_type="msb"):
        if trunc_scaling_impl_type == TruncScalingImplType.MSB:
            return TruncMsbScaling
        elif trunc_scaling_impl_type == TruncScalingImplType.WRAPPER:
            return TruncScalingWrapper
        else:
            raise RuntimeError(f'trunc_scaling_impl_type={trunc_scaling_impl_type} not recognized.')


class SolveTruncIntScalingImplFromEnum(ExtendedInjector):

    @value
    def trunc_int_scaling_impl(restrict_scaling_type):
        if restrict_scaling_type == RestrictValueType.POWER_OF_TWO:
            return PowerOfTwoIntScaling
        else:
            raise RuntimeError(f'restrict_scaling_type={restrict_scaling_type} not recognized.')


class TruncQuantSolver(SolveBitWidthImplFromEnum,
                       SolveTensorQuantFloatToIntImplFromEnum,
                       SolveActScalingImplFromEnum,
                       SolveIntScalingImplFromEnum,
                       SolveScalingStatsOpFromEnum,
                       SolveRestrictScalingImplFromEnum,
                       SolveActScalingInitFromEnum,
                       SolveStatsReduceDimFromEnum,
                       SolveActScalingShape,
                       SolveScalingStatsInputViewShapeImplFromEnum,
                       SolveActScalingPerOutputChannelShape,
                       SolveUpdateStateDictImplFromEnum,
                       SolveInputViewImpl,
                       SolveTruncTensorQuantFromEnum,
                       SolveTruncScalingImplFromEnum,
                       SolveTruncIntScalingImplFromEnum):
    """
    Translate enum directives to truncation-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = TruncQuantProxyFromInjector
