# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List

from dependencies import this
from dependencies import value
import torch
from torch import Tensor

from brevitas.core.bit_width import *
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.function_wrapper import TensorClampSte
from brevitas.core.scaling import *
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
    'SolveParameterScalingShape']


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
        if (bit_width_impl_type == BitWidthImplType.PARAMETER or
                scaling_impl_type == ScalingImplType.AFFINE_STATS or
                scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS or
                scaling_impl_type == ScalingImplType.PARAMETER):
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
