# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.quant_layer import WeightQuantType

from .common import CommonIntAccumulatorAwareWeightQuant
from .common import CommonIntWeightPerChannelQuant
from .common import CommonUintActQuant
from .common import ConstUint8ActQuant

__all__ = [
    "float_espcn", "quant_espcn", "quant_espcn_a2q", "quant_espcn_base", "FloatESPCN", "QuantESPCN"]

IO_DATA_BIT_WIDTH = 8
IO_ACC_BIT_WIDTH = 32


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()


class FloatESPCN(nn.Module):
    """Floating-point version of Efficient Sub-Pixel Convolution Network (ESPCN)"""

    def __init__(self, upscale_factor: int = 3, num_channels: int = 3):
        super(FloatESPCN, self).__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=num_channels * pow(upscale_factor, 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        self.out = nn.ReLU(inplace=True)  # To mirror quant version

        # Initialize weights
        self.apply(weight_init)

    def forward(self, inp: Tensor):
        x = torch.relu(inp)  # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pixel_shuffle(self.conv4(x))
        x = self.out(x)  # To mirror quant version
        return x


class QuantESPCN(FloatESPCN):
    """FINN-Friendly Quantized Efficient Sub-Pixel Convolution Network (ESPCN)"""

    def __init__(
            self,
            upscale_factor: int = 3,
            num_channels: int = 3,
            weight_bit_width: int = 4,
            act_bit_width: int = 4,
            acc_bit_width: int = 32,
            weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant):
        super(QuantESPCN, self).__init__(upscale_factor=upscale_factor)

        # Quantizing the activations to all conv2d layers to unsigned because they
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
            input_bit_width=IO_DATA_BIT_WIDTH,
            input_quant=CommonUintActQuant,
            weight_bit_width=IO_DATA_BIT_WIDTH,
            weight_accumulator_bit_width=IO_ACC_BIT_WIDTH,
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
            bias=True,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=weight_bit_width,
            weight_accumulator_bit_width=acc_bit_width,
            weight_quant=weight_quant)
        # We quantize the weights and input activations of the final layer
        # to 8-bit integers. We do not apply the accumulator constraint to
        # the final convolution layer. FINN does not currently support
        # per-tensor quantization or biases for sub-pixel convolution layers.
        self.conv4 = qnn.QuantConv2d(
            in_channels=32,
            out_channels=num_channels * pow(upscale_factor, 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            input_bit_width=act_bit_width,
            input_quant=CommonUintActQuant,
            weight_bit_width=IO_DATA_BIT_WIDTH,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_scaling_per_output_channel=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)
        # Using a QuantReLU here because we need to read out a uint8 image, but FINN
        # requires a ReLU node to precede an unsigned int quant node
        self.out = qnn.QuantReLU(act_quant=ConstUint8ActQuant, bit_width=IO_DATA_BIT_WIDTH)

        # Initialize weights
        self.apply(weight_init)


def float_espcn(upscale_factor: int, num_channels: int = 3) -> FloatESPCN:
    """ """
    return FloatESPCN(upscale_factor, num_channels=num_channels)


def quant_espcn(
        upscale_factor: int,
        num_channels: int = 3,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        acc_bit_width: int = 32,
        weight_quant: WeightQuantType = CommonIntWeightPerChannelQuant) -> QuantESPCN:
    """ """
    return QuantESPCN(
        upscale_factor=upscale_factor,
        num_channels=num_channels,
        act_bit_width=act_bit_width,
        acc_bit_width=acc_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=weight_quant)


def quant_espcn_a2q(
        upscale_factor: int, weight_bit_width: int, act_bit_width: int, acc_bit_width: int):
    """Integer-quantized FINN-friendly ESPCN model for BSD300 using
    the accumulator-aware weight quantizer"""
    return QuantESPCN(
        upscale_factor=upscale_factor,
        act_bit_width=act_bit_width,
        acc_bit_width=acc_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=CommonIntAccumulatorAwareWeightQuant)


def quant_espcn_base(upscale_factor: int, weight_bit_width: int, act_bit_width: int):
    """Integer-quantized FINN-friendly ESPCN model for BSD300 using
    a vanilla per-channel weight quantizer"""
    return QuantESPCN(
        upscale_factor=upscale_factor,
        act_bit_width=act_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=CommonIntWeightPerChannelQuant)
