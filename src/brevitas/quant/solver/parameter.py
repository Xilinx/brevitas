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
from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.function_wrapper.shape import StatsInputViewShapeImpl
from brevitas.core.scaling import *
from brevitas.core.scaling import ScalingImplType
from brevitas.core.scaling import ScalingPerOutputType
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver.common import *

__all__ = [
    'ScalingConstInit',
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
    def scaling_init_impl(scaling_impl_type):
        if scaling_impl_type == ScalingImplType.CONST:
            return ScalingConstInit
        elif scaling_impl_type == ScalingImplType.PARAMETER:
            return ScalingConstInit
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
            return ParameterFromStatsFromParameterScaling
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
    def scaling_shape(module, group_dim, group_size=None, scaling_per_output=None):
        if scaling_per_output == ScalingPerOutputType.TENSOR:
            return SCALAR_SHAPE
        elif scaling_per_output == ScalingPerOutputType.CHANNEL:
            return this.scaling_per_output_channel_shape
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            assert group_size is not None, "Per Group scaling requires group size"
            assert group_dim is not None, "Per Group scaling requires group dim"
            size = list(module.weight.shape)
            size[group_dim] = (size[group_dim] + group_size - 1) // group_size
            size.insert(group_dim + 1, 1)
            return size

    @value
    def reshaped_scaling_shape(module):
        return module.weight.shape

    @value
    def expanded_scaling_shape(module, group_dim, group_size=None):
        assert group_size is not None, "Per Group scaling requires group size"
        size = list(module.weight.shape)
        size[group_dim] = (size[group_dim] + group_size - 1) // group_size
        size.insert(group_dim + 1, group_size)
        return size

    @value
    def padding(module, group_dim, group_size):
        padding = [0, 0] * len(module.weight.shape)
        size = list(module.weight.shape)
        if size[group_dim] % group_size != 0:
            padding[2 * group_dim] = group_size - size[group_dim] % group_size
        padding = list(reversed(padding))
        return padding

    @value
    def group_dim(module, group_size=None):
        if group_size is not None:
            return 1 if not hasattr(module, 'transposed') or not module.transposed else 0


class SolveInputViewImpl(ExtendedInjector):

    @value
    def input_view_impl(scaling_per_output):
        if scaling_per_output == ScalingPerOutputType.GROUP:
            return StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK
        else:
            return Identity
