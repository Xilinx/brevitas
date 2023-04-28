# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from torch import Tensor
import torch.nn as nn

from brevitas.core.restrict_val import RestrictValueType
import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant.fixed_point import Int8AccumulatorAwareWeightQuant
from brevitas.quant.fixed_point import Int8WeightNormL2PerChannelFixedPoint


class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloat):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloat):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP


class QuantNearestNeighborConvolution(nn.Module):
    """Quantized nearest neighbor resize convolution"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Optional[int] = 3,
            stride: Optional[int] = 1,
            padding: Optional[int] = 1,
            upscale_factor: Optional[int] = 3,
            signed_act: Optional[bool] = False,
            bias: Optional[bool] = True,
            weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant,
            act_bit_width: Optional[int] = 8,
            weight_bit_width: Optional[int] = 8):
        super().__init__()

        act_quant = CommonIntActQuant if signed_act else CommonUintActQuant

        self.upscale_factor = upscale_factor
        self.input_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=True, bit_width=act_bit_width)
        self.interp = qnn.QuantUpsamplingNearest2d(
            scale_factor=upscale_factor, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
            input_quant=None,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.conv.cache_inference_quant_inp = True  # needed for post-training evaluation of accumulator bit-width

    def forward(self, inp: Tensor) -> Tensor:
        return self.conv(self.interp(self.input_quant(inp)))
