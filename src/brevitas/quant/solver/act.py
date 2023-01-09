# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import torch
from torch import Tensor, nn
from brevitas.core.quant import RescalingIntQuant, TernaryQuant, ClampedBinaryQuant
from brevitas.core.scaling import ParameterScaling, ConstScaling, SCALAR_SHAPE
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling, RuntimeStatsScaling
from brevitas.proxy.utils import ConvertRuntimeStatsToParameter
from brevitas.quant.solver.common import *
from brevitas.inject import ExtendedInjector, value, this
from brevitas.inject.enum import ScalingImplType, QuantType
from brevitas.proxy import ActQuantProxyFromInjector


class MinMaxScalingInit:

    def __init__(self, min_val: float, max_val: float):
        self.scaling_init = torch.tensor(max(abs(float(min_val)), abs(float(max_val))))

    def __call__(self):
        return self.scaling_init


class SolveActScalingImplFromEnum(SolveAffineRescalingFromEnum):

    @value
    def scaling_impl(scaling_impl_type):
        if scaling_impl_type is None:
            return None
        elif scaling_impl_type == ScalingImplType.PARAMETER:
            return ParameterScaling
        elif scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            return ParameterFromRuntimeStatsScaling
        elif scaling_impl_type == ScalingImplType.CONST:
            return ConstScaling
        elif scaling_impl_type == ScalingImplType.STATS:
            return RuntimeStatsScaling
        elif scaling_impl_type == ScalingImplType.AFFINE_STATS:
            return RuntimeStatsScaling
        elif scaling_impl_type == ScalingImplType.HE:
            raise RuntimeError(f"{scaling_impl_type} not supported.")
        else:
            raise RuntimeError(f"{scaling_impl_type} not recognized.")


class SolveActTensorQuantFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return RescalingIntQuant
        elif quant_type == QuantType.TERNARY:
            return TernaryQuant
        elif quant_type == QuantType.BINARY:
            return ClampedBinaryQuant
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class SolveActScalingInitFromEnum(ExtendedInjector):

    @value
    def scaling_init(scaling_init_impl):
        scaling_init = scaling_init_impl()
        if isinstance(scaling_init, Tensor):
            return scaling_init.detach()
        else:
            return torch.tensor(scaling_init)

    @value
    def scaling_init_impl(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.CONST:
            return MinMaxScalingInit
        elif scaling_impl_type == ScalingImplType.PARAMETER:
            return MinMaxScalingInit
        else:
            return None

    @value
    def min_val(signed):
        if not signed:
            return 0.
        else:
            return None


class SolveActScalingShape(ExtendedInjector):

    @value
    def scaling_shape(scaling_per_output_channel):
        # this pattern of returning this.something allows to resolve scaling_output_channel_shape
        # only when scaling_per_output_channel is True
        if scaling_per_output_channel:
            return this.per_channel_broadcastable_shape
        else:
            return SCALAR_SHAPE


class SolveActScalingPerOutputChannelShape(ExtendedInjector):

    @value
    def scaling_per_output_channel_shape(per_channel_broadcastable_shape):
        return per_channel_broadcastable_shape


class SolveUpdateStateDictImplFromEnum(ExtendedInjector):

    @value
    def update_state_dict_impl(scaling_impl_type):
        if (scaling_impl_type == ScalingImplType.PARAMETER
                or scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS):
            return ConvertRuntimeStatsToParameter
        else:
            return None


class ActQuantSolver(
        SolveActTensorQuantFromEnum,
        SolveActScalingImplFromEnum,
        SolveIntScalingImplFromEnum,
        SolveBitWidthImplFromEnum,
        SolveTensorQuantFloatToIntImplFromEnum,
        SolveScalingStatsOpFromEnum,
        SolveRestrictScalingImplFromEnum,
        SolveActScalingInitFromEnum,
        SolveStatsReduceDimFromEnum,
        SolveActScalingShape,
        SolveScalingStatsInputViewShapeImplFromEnum,
        SolveActScalingPerOutputChannelShape,
        SolveUpdateStateDictImplFromEnum):
    """
    Translate enum directives to activation-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = ActQuantProxyFromInjector



