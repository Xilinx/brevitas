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

import math
from typing import List

from dependencies import value, this
import torch
from torch import Tensor

from brevitas.core.function_wrapper import TensorClamp, TensorClampSte
from brevitas.core.scaling import *
from brevitas.core.bit_width import *
from brevitas.core.scaling import ScalingImplType
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver.common import *


__all__ = [
    'ScalingConstInit',
    'ParameterFromStatsScalingInit',
    'HeScalingInit',
    'SolveParameterTensorClampImplFromEnum',
    'SolveParameterScalingInitFromEnum',
    'SolveParameterScalingImplFromEnum',
    'SolveParameterScalingShape'
]


class ScalingConstInit:

    def __init__(self, scaling_const):
        self.scaling_const = scaling_const

    def __call__(self):
        return self.scaling_const


class ParameterFromStatsScalingInit:

    def __init__(self, parameter_stats_scaling_init_impl):
        self.init_impl = parameter_stats_scaling_init_impl

    def __call__(self):
        return self.init_impl(torch.tensor(0.0))


class HeScalingInit:

    def __init__(self, tracked_parameter_list: List[torch.nn.Parameter]):
        self.tracked_parameter_list = tracked_parameter_list

    def __call__(self):
        scaling_init = 0.0
        # takes average of He scaling over parameter list
        for param in self.tracked_parameter_list:
            two_dim_param = param.view(param.shape[0], -1)
            scaling_init += math.sqrt(2.0 / two_dim_param.shape[1])
        scaling_init /= len(self.tracked_parameter_list)
        return torch.tensor(scaling_init)



class SolveParameterTensorClampImplFromEnum(ExtendedInjector):

    @value
    def tensor_clamp_impl(bit_width_impl_type, scaling_impl_type):
        if (bit_width_impl_type == BitWidthImplType.PARAMETER
                or scaling_impl_type == ScalingImplType.AFFINE_STATS
                or scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS
                or scaling_impl_type == ScalingImplType.PARAMETER):
            return TensorClamp
        else:
            return TensorClampSte


class SolveParameterScalingInitFromEnum(ExtendedInjector):

    @value
    def scaling_init(scaling_init_impl):
        scaling_init = scaling_init_impl()
        if isinstance(scaling_init, Tensor):
            return scaling_init.detach()
        else:
            return torch.tensor(scaling_init)

    @value
    def parameter_stats_scaling_init_impl(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            return StatsFromParameterScaling
        else:
            return None

    @value
    def scaling_init_impl(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.CONST:
            return ScalingConstInit
        elif scaling_impl_type == ScalingImplType.PARAMETER:
            return ScalingConstInit
        elif scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            return ParameterFromStatsScalingInit
        elif scaling_impl_type == ScalingImplType.HE:
            return HeScalingInit
        else:
            return None


class SolveParameterScalingImplFromEnum(SolveAffineRescalingFromEnum):

    @value
    def scaling_impl(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.PARAMETER:
            return ParameterScaling
        elif scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            return ParameterScaling
        elif scaling_impl_type == ScalingImplType.CONST:
            return ConstScaling
        elif scaling_impl_type == ScalingImplType.HE:
            return ConstScaling
        elif scaling_impl_type == ScalingImplType.STATS:
            return StatsFromParameterScaling
        elif scaling_impl_type == ScalingImplType.AFFINE_STATS:
            return StatsFromParameterScaling
        else:
            raise RuntimeError(f"{scaling_impl_type} not recognized.")


class SolveParameterScalingShape(ExtendedInjector):

    @value
    def scaling_shape(scaling_per_output_channel):
        # this pattern of returning this.something allows to resolve scaling_output_channel_shape
        # only when scaling_per_output_channel is True
        if scaling_per_output_channel:
            return this.scaling_per_output_channel_shape
        else:
            return SCALAR_SHAPE