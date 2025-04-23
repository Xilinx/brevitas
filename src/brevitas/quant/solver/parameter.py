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
    def scaling_impl(scaling_impl_type=None):
        # Needed for no-scale minifloat quantization
        if scaling_impl_type is None:
            return None

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
    def scaling_shape(scaling_per_output, expanded_groupwise_shape=None, group_dim=None):
        if scaling_per_output == ScalingPerOutputType.TENSOR:
            return SCALAR_SHAPE
        elif scaling_per_output == ScalingPerOutputType.CHANNEL:
            return this.scaling_per_output_channel_shape
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
            assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
            assert group_dim is not None, "Per Group scaling not correctly configured"
            size = list(expanded_groupwise_shape)
            size[group_dim + 1] = 1
            return tuple(size)

    @value
    def reshaped_groupwise_shape(expanded_groupwise_shape, group_dim, group_size):
        new_shape = list(expanded_groupwise_shape)
        del new_shape[group_dim + 1]  # delete the group_size shape
        # Expand the group_dim shape, accounting for padding
        new_shape[group_dim] = new_shape[group_dim] * group_size
        return new_shape

    @value
    def expanded_groupwise_shape(tracked_parameter_list, group_dim, group_size=None):
        # expanded_groupwise_shape will be called always to create scaling_shape, but it is only needed
        # for groupwise quantization. All other groupwise shape infos are derived from this.

        # If conditions do not allow for groupwise quantization, early exit and return None
        if group_size is None:
            return

        # If group_size is specified and shared quantization is used, raise an error.
        assert len(tracked_parameter_list) == 1, "Shared groupwise quantization is not currently supported"

        weight_shape = tracked_parameter_list[0].shape
        size = list(weight_shape)
        size[group_dim] = (size[group_dim] + group_size - 1) // group_size
        size.insert(group_dim + 1, group_size)
        return tuple(size)

    @value
    def group_dim(module, group_size=None):
        # group_dim will be called always to create scaling_shape, but it is only needed
        # for groupwise quantization.
        if group_size is not None:
            return 1 if not hasattr(module, 'transposed') or not module.transposed else 0


class SolveInputViewImpl(ExtendedInjector):

    @value
    def input_view_impl(scaling_per_output):
        if scaling_per_output == ScalingPerOutputType.GROUP:
            return StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK
        else:
            return Identity
