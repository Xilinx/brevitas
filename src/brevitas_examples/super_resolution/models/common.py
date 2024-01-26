# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from torch import Tensor
import torch.nn as nn

from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType
from brevitas.quant import Int8AccumulatorAwareWeightQuant
from brevitas.quant import Int8AccumulatorAwareZeroCenterWeightQuant
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat


class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntAccumulatorAwareWeightQuant(Int8AccumulatorAwareWeightQuant):
    """A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance"""
    restrict_scaling_impl = FloatRestrictValue  # backwards compatibility
    bit_width = None


class CommonIntAccumulatorAwareZeroCenterWeightQuant(Int8AccumulatorAwareZeroCenterWeightQuant):
    """A2Q+: Improving Accumulator-Aware Weight Quantization"""
    bit_width = None


class CommonIntActQuant(Int8ActPerTensorFloat):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloat):
    """Common unsigned act quantizer with bit-width set to None so that it's forced to be
    specified by each layer"""
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP


class ConstUint8ActQuant(CommonUintActQuant):
    """8-bit unsigned integer activation quantizer with constant unit scaling factor, used
    by the models to quantize outputs into the image space"""
    scaling_impl_type = ScalingImplType.CONST
    scaling_init = 1.


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
            acc_bit_width: Optional[int] = 32,
            act_bit_width: Optional[int] = 8,
            weight_bit_width: Optional[int] = 8):
        super().__init__()

        # Using unsigned int activation quantization if the preceding layer has
        # a non-negative range (e.g., following a ReLU activation function)
        act_quant = CommonIntActQuant if signed_act else CommonUintActQuant

        self.upscale_factor = upscale_factor
        # Need to have the quantization node before the nearest neighbor upsampling node
        # for FINN compatibility since the FINN compiler will streamline the quantization
        # node with the preceding monotonic activation function. In the case of ESPCN, this
        # is a ReLU. We need to return the QuantTensor though so that the conv2d is aware
        # of the input bit-width for accumulator-aware quantization (A2Q). For more discussion
        # on this, see https://arxiv.org/abs/2301.13376.
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
            weight_accumulator_bit_width=acc_bit_width,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)

    def forward(self, inp: Tensor) -> Tensor:
        return self.conv(self.interp(self.input_quant(inp)))
