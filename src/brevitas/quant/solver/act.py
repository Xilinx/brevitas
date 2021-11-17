# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
    def scaling_per_output_channel(scaling_per_channel):
        """Alternative syntax"""
        return scaling_per_channel

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



