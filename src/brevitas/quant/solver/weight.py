# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.quant import *
from brevitas.core.quant import QuantType
from brevitas.inject import ExtendedInjector
from brevitas.inject import this
from brevitas.inject import value
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.solver.common import *
from brevitas.quant.solver.parameter import *

__all__ = [
    'SolveWeightTensorQuantFromEnum',
    'SolveWeightScalingStatsInputDimsFromModule',
    'SolveWeightScalingPerOutputChannelShapeFromModule',
    'WeightQuantSolver']


class SolveWeightTensorQuantFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return RescalingIntQuant
        elif quant_type == QuantType.TERNARY:
            return TernaryQuant
        elif quant_type == QuantType.BINARY:
            return BinaryQuant
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class SolveWeightScalingPerOutputChannelShapeFromModule(ExtendedInjector):

    @value
    def out_channels(module):
        if isinstance(module, tuple):
            assert all(m.out_channels == module[0].out_channels for m in module)
            module = module[0]
        return module.out_channels

    @value
    def weight_ndims(module):
        if isinstance(module, tuple):
            assert all(len(m.weight.size()) == len(module[0].weight.size()) for m in module)
            module = module[0]
        return len(module.weight.size())

    @value
    def scaling_per_output_channel_shape(weight_ndims, output_channel_dim, out_channels):
        shape = [1] * weight_ndims
        shape[output_channel_dim] = out_channels
        return tuple(shape)


class SolveWeightScalingStatsInputDimsFromModule(ExtendedInjector):

    #  during per-channel quantization weights are always permuted and reshaped first
    #  such that output channels are dim 0 and the remaining features are dim 1,
    #  along which we concatenate
    @value
    def scaling_stats_input_concat_dim(scaling_per_output_channel):
        if scaling_per_output_channel:
            return 1
        else:
            return 0

    @value
    def permute_dims(module, output_channel_dim):
        if output_channel_dim != 0:
            if isinstance(module, tuple):
                assert all(len(m.weight.shape) == len(module[0].weight.shape) for m in module)
                module = module[0]
            dims = list(range(0, len(module.weight.shape)))
            dims[0], dims[output_channel_dim] = dims[output_channel_dim], dims[0]
            return tuple(dims)
        else:
            return None

    @value
    def output_channel_dim(module):
        if isinstance(module, tuple):
            assert all(m.output_channel_dim == module[0].output_channel_dim for m in module)
            module = module[0]
        return module.output_channel_dim


class WeightQuantSolver(SolveWeightScalingStatsInputDimsFromModule,
                        SolveScalingStatsInputViewShapeImplFromEnum,
                        SolveStatsReduceDimFromEnum,
                        SolveScalingStatsOpFromEnum,
                        SolveBitWidthImplFromEnum,
                        SolveTensorQuantFloatToIntImplFromEnum,
                        SolveRestrictScalingImplFromEnum,
                        SolveIntScalingImplFromEnum,
                        SolveParameterScalingImplFromEnum,
                        SolveParameterTensorClampImplFromEnum,
                        SolveParameterScalingInitFromEnum,
                        SolveParameterScalingShape,
                        SolveWeightScalingPerOutputChannelShapeFromModule,
                        SolveWeightTensorQuantFromEnum):
    """
    Translate enum and shape directives to weight-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = WeightQuantProxyFromInjector
