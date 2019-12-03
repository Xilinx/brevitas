#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batchnorm_reimpl.py
# Author : acgtyrant
# Date   : 11/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from typing import Optional

import torch
import torch.nn as nn

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl, StatsOp
from brevitas.nn.quant_layer import SCALING_MIN_VAL
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy, OVER_BATCH_OVER_CHANNELS_SHAPE
from .quant_layer import QuantLayer

__all__ = ['ScaleBias', 'QuantScaleBias']


class ScaleBias(nn.Module):

    def __init__(self, num_features):
        super(ScaleBias, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.weight + self.bias


class QuantScaleBias(QuantLayer, ScaleBias):

    def __init__(self,
                 num_features,
                 bias_quant_type: QuantType = QuantType.FP,
                 bias_narrow_range: bool = False,
                 bias_bit_width: int = None,
                 weight_quant_type: QuantType = QuantType.FP,
                 weight_quant_override: nn.Module = None,
                 weight_narrow_range: bool = False,
                 weight_scaling_override: Optional[nn.Module] = None,
                 weight_bit_width: int = 32,
                 weight_scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
                 weight_scaling_const: Optional[float] = None,
                 weight_scaling_stats_op: StatsOp = StatsOp.MAX,
                 weight_scaling_per_output_channel: bool = False,
                 weight_restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                 weight_scaling_stats_sigma: float = 3.0,
                 weight_scaling_min_val: float = SCALING_MIN_VAL,
                 compute_output_scale: bool = False,
                 compute_output_bit_width: bool = False,
                 return_quant_tensor: bool = False):
        QuantLayer.__init__(self,
                            compute_output_scale=compute_output_scale,
                            compute_output_bit_width=compute_output_bit_width,
                            return_quant_tensor=return_quant_tensor)
        ScaleBias.__init__(self, num_features)

        if bias_quant_type != QuantType.FP and not self.compute_output_scale:
            raise Exception("Quantizing bias requires to compute output scale")
        if bias_quant_type != QuantType.FP and bias_bit_width is None and not self.compute_output_bit_width:
            raise Exception("Quantizing bias requires a bit-width, either computed or defined")

        if weight_quant_override is not None:
            self.weight_quant = weight_quant_override
            self.weight_quant.add_tracked_parameter(self.weight)
        else:
            weight_scaling_stats_input_concat_dim = 1
            if weight_scaling_stats_op == StatsOp.MAX_AVE:
                assert not weight_scaling_per_output_channel
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
                weight_scaling_shape = SCALING_SCALAR_SHAPE
                weight_scaling_stats_reduce_dim = None
            else:
                if weight_scaling_per_output_channel:
                    weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
                    weight_scaling_shape = (num_features, 1)
                    weight_scaling_stats_reduce_dim = 1
                else:
                    weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_TENSOR
                    weight_scaling_shape = SCALING_SCALAR_SHAPE
                    weight_scaling_stats_reduce_dim = None

            self.weight_quant = WeightQuantProxy(bit_width=weight_bit_width,
                                                 quant_type=weight_quant_type,
                                                 narrow_range=weight_narrow_range,
                                                 scaling_override=weight_scaling_override,
                                                 restrict_scaling_type=weight_restrict_scaling_type,
                                                 scaling_const=weight_scaling_const,
                                                 scaling_stats_op=weight_scaling_stats_op,
                                                 scaling_impl_type=weight_scaling_impl_type,
                                                 scaling_stats_reduce_dim=weight_scaling_stats_reduce_dim,
                                                 scaling_shape=weight_scaling_shape,
                                                 bit_width_impl_type=BitWidthImplType.CONST,
                                                 bit_width_impl_override=None,
                                                 restrict_bit_width_type=RestrictValueType.INT,
                                                 min_overall_bit_width=None,
                                                 max_overall_bit_width=None,
                                                 tracked_parameter_list_init=self.weight,
                                                 ternary_threshold=None,
                                                 scaling_stats_input_view_shape_impl=weight_stats_input_view_shape_impl,
                                                 scaling_stats_input_concat_dim=weight_scaling_stats_input_concat_dim,
                                                 scaling_stats_sigma=weight_scaling_stats_sigma,
                                                 scaling_min_val=weight_scaling_min_val,
                                                 override_pretrained_bit_width=None)
        self.bias_quant = BiasQuantProxy(quant_type=bias_quant_type,
                                         narrow_range=bias_narrow_range,
                                         bit_width=bias_bit_width)

    def forward(self, quant_tensor):
        output_scale = None
        output_bit_width = None

        input_tensor, input_scale, input_bit_width = self.unpack_input(quant_tensor)
        quant_weight, quant_weight_scale, quant_weight_bit_width = self.weight_quant(self.weight.view(-1, 1))

        if self.compute_output_bit_width:
            assert input_bit_width is not None
            output_bit_width = input_bit_width + quant_weight_bit_width
        if self.compute_output_scale:
            assert input_scale is not None
            output_scale = input_scale * quant_weight_scale

        # if bias_bit_width is not None, input_bit_width is ignored
        quant_bias, _, quant_bias_bit_width = self.bias_quant(self.bias, output_scale, input_bit_width)

        quant_weight = quant_weight.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
        quant_bias = quant_bias.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
        output = input_tensor * quant_weight + quant_bias

        if self.compute_output_bit_width and quant_bias_bit_width is not None:
            output_bit_width = torch.where(quant_bias_bit_width > output_bit_width,
                                           quant_bias_bit_width,
                                           output_bit_width)

        return self.pack_output(output, output_scale, output_bit_width)




