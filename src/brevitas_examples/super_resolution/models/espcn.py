# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType

from .common import CommonIntWeightPerChannelQuant
from .common import CommonUintActQuant
from .common import Int8AccumulatorAwareWeightQuant
from .common import Int8WeightPerTensorFloat
from .common import QuantNearestNeighborConvolution

__all__ = [
    "float_espcn",
    "quant_espcn_w8a8",
    "quant_espcn_w4a4",
    "quant_espcn_finn_a2q_w4a4_14b",
    "quant_espcn_finn_a2q_w4a4_32b"]

IO_BIT_WIDTH = 8


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolution Network (ESPCN) as proposed by Shi et al. (2016)
    in `Real-time single image and video super-resolution using an efficient sub-pixel
    convolutional neural network.`"""

    def __init__(self, in_channels: int, upscale_factor: int):
        super(ESPCN, self).__init__()
        out_channels = in_channels * pow(upscale_factor, 2)
        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.apply(weight_init)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


class QuantESPCN(ESPCN):
    """Quantized ESPCN derived from floating-point ESPCN class"""

    def __init__(
            self,
            in_channels: int,
            upscale_factor: int,
            act_bit_width: int,
            weight_bit_width: int,
            weight_quant: WeightQuantType):
        super().__init__(in_channels, upscale_factor)
        out_channels = in_channels * pow(upscale_factor, 2)
        self.conv1 = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            input_bit_width=IO_BIT_WIDTH,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.conv2 = qnn.QuantConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.conv3 = qnn.QuantConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.conv4 = qnn.QuantConv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)


def float_espcn(upscale_factor: int) -> ESPCN:
    """Floating-point implementation of the efficient sub-pixel convolution network"""
    return ESPCN(1, upscale_factor)


def quant_espcn_w8a8(upscale_factor: int):
    """4-bit integer quantized ESPCN model for BSD300 using common
    integer weight quantization with per-tensor scales"""
    return QuantESPCN(1, upscale_factor, 8, 8, Int8WeightPerTensorFloat)


def quant_espcn_w4a4(upscale_factor: int):
    """4-bit integer quantized ESPCN model for BSD300 using common
    integer weight quantization with per-tensor scales"""
    return QuantESPCN(1, upscale_factor, 4, 4, Int8WeightPerTensorFloat)


class QuantESPCNV2(nn.Module):
    """FINN-Friendly Quantized Efficient Sub-Pixel Convolution Network (ESPCN) as
    proposed in Colbert et al. (2023) - `Quantized Neural Networks for Low-Precision
    Accumulation with Guaranteed Overflow Avoidance`."""
    def __init__(
            self,
            upscale_factor: int = 3,
            num_channels: int = 3,
            weight_bit_width: int = 4,
            act_bit_width: int = 4,
            acc_bit_width: int = 32,
            weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant):
        super(QuantESPCNV2, self).__init__()

        # Quantizing the input of conv2d layers to unsigned because they
        # are all preceded by ReLUs, which have non-negative ranges
        self.conv1 = qnn.QuantConv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            input_bit_width=IO_BIT_WIDTH,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_accumulator_bit_width=acc_bit_width,
            weight_quant=weight_quant)
        self.conv2 = qnn.QuantConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_accumulator_bit_width=acc_bit_width,
            weight_quant=weight_quant)
        self.conv3 = qnn.QuantConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_accumulator_bit_width=acc_bit_width,
            weight_quant=weight_quant)
        # Not applying accumulator constraint to the final convolution layer
        self.conv4 = QuantNearestNeighborConvolution(
            in_channels=32,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            upscale_factor=upscale_factor,
            bias=True,
            signed_act=False,
            act_bit_width=act_bit_width,
            weight_quant=weight_quant,
            weight_bit_width=weight_bit_width)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        # Using a QuantReLU here because we need to read out a uint8 image, but FINN
        # requires a ReLU node to precede an unsigned int quant node
        self.out = qnn.QuantReLU(act_quant=CommonUintActQuant, bit_width=IO_BIT_WIDTH)

        self.apply(weight_init)

    def forward(self, inp: Tensor):
        x = torch.relu(inp)  # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.out(self.conv4(x))
        return x


def quant_espcn_finn_a2q_w4a4_14b(upcsale_factor: int):
    """4-bit integer quantized FINN-friendly ESPCN model for BSD300 using
    integer accumulator-aware weight quantizer for a 14-bit accumulator"""
    return QuantESPCNV2(
        upscale_factor=upcsale_factor,
        num_channels=1,
        act_bit_width=4,
        acc_bit_width=14,
        weight_bit_width=4,
        weight_quant=Int8AccumulatorAwareWeightQuant)


def quant_espcn_finn_a2q_w4a4_32b(upscale_factor: int):
    """4-bit integer quantized FINN-friendly ESPCN model for BSD300 using
    integer accumulator-aware weight quantizer for a 32-bit accumulator"""
    return QuantESPCNV2(
        upscale_factor=upscale_factor,
        num_channels=1,
        act_bit_width=4,
        acc_bit_width=32,
        weight_bit_width=4,
        weight_quant=Int8AccumulatorAwareWeightQuant)
