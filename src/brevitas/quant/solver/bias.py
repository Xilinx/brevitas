# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.quant import PrescaledRestrictIntQuant
from brevitas.core.quant import RescalingIntQuant
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import QuantType
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.quant.solver.common import *
from brevitas.quant.solver.parameter import *
from brevitas.quant.solver.parameter import SolveInputViewImpl

__all__ = [
    'BiasQuantSolver',
    'SolveBiasTensorQuantFromEnum',
    'SolveBiasScalingStatsInputConcatDimFromModule',
    'SolveBiasBitWidthImplFromEnum',
    'SolveBiasScalingPerOutputChannelShapeFromModule']


class SolveBiasScalingStatsInputConcatDimFromModule(ExtendedInjector):
    scaling_stats_input_concat_dim = 0  # bias has only 1 dimension by definition


class SolveBiasScalingPerOutputChannelShapeFromModule(ExtendedInjector):

    @value
    def scaling_per_output_channel_shape(module):
        if isinstance(module, tuple):
            assert all(m.out_channels == module[0].out_channels for m in module)
            module = module[0]
        return (module.out_channels,)


class SolveBiasTensorQuantFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_quant(quant_type, requires_input_scale):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            if requires_input_scale:
                return PrescaledRestrictIntQuant
            else:
                return RescalingIntQuant
        elif quant_type == QuantType.TERNARY:
            raise RuntimeError(f'{quant_type} not supported.')
        elif quant_type == QuantType.BINARY:
            raise RuntimeError(f'{quant_type} not supported.')
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class BiasQuantSolver(SolveScalingStatsInputViewShapeImplFromEnum,
                      SolveParameterScalingShape,
                      SolveStatsReduceDimFromEnum,
                      SolveScalingStatsOpFromEnum,
                      SolveTensorQuantFloatToIntImplFromEnum,
                      SolveRestrictScalingImplFromEnum,
                      SolveIntScalingImplFromEnum,
                      SolveParameterScalingImplFromEnum,
                      SolveParameterTensorClampImplFromEnum,
                      SolveParameterScalingInitFromEnum,
                      SolveBitWidthImplFromEnum,
                      SolveBiasScalingPerOutputChannelShapeFromModule,
                      SolveBiasScalingStatsInputConcatDimFromModule,
                      SolveBiasTensorQuantFromEnum,
                      SolveDtypeDeviceFromTrackedParameterList,
                      SolveInputViewImpl):
    """
    Translate enum directives to bias-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = BiasQuantProxyFromInjector
