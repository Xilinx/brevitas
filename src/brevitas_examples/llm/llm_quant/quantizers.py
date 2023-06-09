"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from torch import nn

from brevitas.core.scaling import StatsFromParameterScaling
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.inject import this
from brevitas.inject import value
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

from .quant_blocks import *


class IntWeightSymmetricGroupQuant(Int8WeightPerChannelFloat):
    """
    Block / group / vector signed symmetric weight quantizer with float scales.
    We inherit from a per-channel quantizer to re-use some underlying machinery.
    """

    @value
    def expanded_scaling_shape(module, block_size):
        if isinstance(module, nn.Conv2d):
            return module.weight.size(0), module.weight.size(1) // block_size, block_size, module.weight.size(2), module.weight.size(3)
        elif isinstance(module, nn.Linear):
            return module.weight.size(0), module.weight.size(1) // block_size, block_size
        else:
            raise RuntimeError("Module not supported.")

    @value
    def scaling_shape(module, block_size):
        if isinstance(module, nn.Conv2d):
            return module.weight.size(0), module.weight.size(1) // block_size, 1, module.weight.size(2), module.weight.size(3)
        elif isinstance(module, nn.Linear):
            return module.weight.size(0), module.weight.size(1) // block_size, 1
        else:
            raise RuntimeError("Module not supported.")

    @value
    def reshaped_scaling_shape(module):
        return module.weight.shape

    scaling_input_shape = this.expanded_scaling_shape
    scaling_stats_input_view_shape_impl = OverSubChannelBlockView
    scaling_impl = ExpandReshapeScalingWrapper
    wrapped_scaling_impl = StatsFromParameterScaling
    scaling_stats_impl = AbsMaxKeepDim
    # scale is converted to a parameter right away
    scaling_impl_type = 'parameter_from_stats'
    stats_reduce_dim = 2
    # Set bit_width and block size externally
    bit_width = None
    block_size = None


class ShiftedUintWeightAsymmetricGroupQuant(IntWeightSymmetricGroupQuant):
    """
    Block / group / vector signed asymmetric weight quantizer with float scales and zero-points.
    """
    zero_point_input_shape = this.scaling_input_shape
    reshaped_zero_point_shape = this.reshaped_scaling_shape
    zero_point_shape = this.scaling_shape
    expanded_zero_point_shape = this.expanded_scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
    zero_point_stats_input_concat_dim = 0
    zero_point_impl = ExpandReshapeZeroPointWrapper
    zero_point_stats_impl = NegativeMinOrZeroKeepDim
    scaling_stats_impl = AbsMinMaxKeepDim
    # zero-point is converted to a parameter right away
    wrapped_zero_point_impl = ParameterFromStatsFromParameterZeroPoint
    quantize_zero_point = False
    signed = False


class Int8ActPerRowFloat(Int8ActPerTensorFloat):
    scaling_per_output_channel = True


class Int8ActPerRowFloatMSE(Int8ActPerTensorFloatMSE):
    scaling_per_output_channel = True
