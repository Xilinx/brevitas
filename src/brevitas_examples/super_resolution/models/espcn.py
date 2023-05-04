# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType

from .common import CommonIntWeightPerChannelQuant
from .common import CommonUintActQuant
from .common import CommonIntAccumulatorAwareWeightQuant
from .common import QuantNearestNeighborConvolution

__all__ = ["float_espcn", "quant_espcn", "quant_espcn_a2q", "quant_espcn_base"]

IO_BIT_WIDTH = 8


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()


class FloatESPCN(nn.Module):
    """Floating-point version of FINN-Friendly Quantized Efficient Sub-Pixel Convolution
    Network (ESPCN) as used in Colbert et al. (2023) - `Quantized Neural Networks for
    Low-Precision Accumulation with Guaranteed Overflow Avoidance`."""
    def __init__(
            self,
            upscale_factor: int = 3,
            num_channels: int = 1):
        super(FloatESPCN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv4 = nn.Sequential()
        self.conv4.add_module(
            "interp",
            nn.UpsamplingNearest2d(scale_factor=upscale_factor))
        self.conv4.add_module(
            "conv",
            nn.Conv2d(
                in_channels=32,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        self.out = nn.ReLU(inplace=True) # To mirror quant version

        # Initialize weights
        self.apply(weight_init)

    def forward(self, inp: Tensor):
        x = torch.relu(inp)  # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.out(self.conv4(x))
        return x


class QuantESPCN(FloatESPCN):
    """FINN-Friendly Quantized Efficient Sub-Pixel Convolution Network (ESPCN) as
    used in Colbert et al. (2023) - `Quantized Neural Networks for Low-Precision
    Accumulation with Guaranteed Overflow Avoidance`."""
    def __init__(
            self,
            upscale_factor: int = 3,
            num_channels: int = 3,
            weight_bit_width: int = 4,
            act_bit_width: int = 4,
            acc_bit_width: int = 32,
            weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant):
        super(QuantESPCN, self).__init__()

        # Quantizing the input of conv2d layers to unsigned because they
        # are all preceded by ReLUs, which have non-negative ranges

        # Quantizing input quant conv2d to use 8-bit inputs and weights with
        # an accumulator size of 32 bits
        self.conv1 = qnn.QuantConv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            input_bit_width=IO_BIT_WIDTH,
            input_quant=CommonUintActQuant,
            weight_bit_width=IO_BIT_WIDTH,
            weight_accumulator_bit_width=32,
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
        # Quantizing the weights and input activations to 8-bit integers
        # and not applying accumulator constraint to the final convolution
        # layer (i.e., accumulator_bit_width=32).
        self.conv4 = QuantNearestNeighborConvolution(
            in_channels=32,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            upscale_factor=upscale_factor,
            bias=True,
            signed_act=False,
            act_bit_width=IO_BIT_WIDTH,
            weight_quant=weight_quant,
            weight_bit_width=IO_BIT_WIDTH)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        # Using a QuantReLU here because we need to read out a uint8 image, but FINN
        # requires a ReLU node to precede an unsigned int quant node
        self.out = qnn.QuantReLU(act_quant=CommonUintActQuant, bit_width=IO_BIT_WIDTH)

        # Initialize weights
        self.apply(weight_init)


def float_espcn(upscale_factor: int, num_channels: int = 1) -> FloatESPCN:
    """ """
    return FloatESPCN(upscale_factor, num_channels=num_channels)


def quant_espcn(
        upcsale_factor: int,
        num_channels: int = 1,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        acc_bit_width: int = 8,
        weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant) -> QuantESPCN:
    """ """
    return QuantESPCN(
        upscale_factor=upcsale_factor,
        num_channels=num_channels,
        act_bit_width=act_bit_width,
        acc_bit_width=acc_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=weight_quant)


def quant_espcn_a2q(upscale_factor: int, weight_bit_width: int, act_bit_width: int, acc_bit_width: int):
    """Integer-quantized FINN-friendly ESPCN model for BSD300 using
    the accumulator-aware weight quantizer"""
    return QuantESPCN(
        upscale_factor=upscale_factor,
        num_channels=1,
        act_bit_width=act_bit_width,
        acc_bit_width=acc_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=CommonIntAccumulatorAwareWeightQuant)


def quant_espcn_base(upscale_factor: int, weight_bit_width: int, act_bit_width: int):
    """Integer-quantized FINN-friendly ESPCN model for BSD300 using
    a vanilla per-channel weight quantizer"""
    return QuantESPCN(
        upscale_factor=upscale_factor,
        num_channels=1,
        act_bit_width=act_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=CommonIntWeightPerChannelQuant)