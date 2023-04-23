# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

import brevitas.nn as qnn

from .common import QuantNearestNeighborConvolution
from .common import CommonUintActQuant
from .common import CommonIntWeightPerChannelQuant

__all__ = [
    "quant_espcn_x3_v1_4b",
    "quant_espcn_x3_v2_4b",
    "quant_espcn_x3_v3_4b"]

IO_BIT_WIDTH = 8


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()


class QuantESPCN(nn.Module):
    """ Quantized Efficient Sub-Pixel Convolution Network (ESPCN) """
    def __init__(
        self,
        upscale_factor: int = 3,
        num_channels: int = 3,
        weight_bit_width: int = 4,
        act_bit_width: int = 4,
        weight_quant = CommonIntWeightPerChannelQuant):
        super(QuantESPCN, self).__init__()

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
        self.out = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=IO_BIT_WIDTH)

        self.apply(weight_init)

    def forward(self, inp: Tensor):
        x = torch.relu(inp) # Adding for finn-onnx compatability
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.out(self.conv4(x))
        return x


def quant_espcn_v1(
    num_channels: int,
    upscale_factor: int,
    weight_bit_width: int,
    act_bit_width: int,
    weight_quant) -> QuantESPCN:
    model = QuantESPCN(
        upscale_factor=upscale_factor,
        num_channels=num_channels,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        weight_quant=weight_quant)
    return model


def quant_espcn_x3_v1_4b():
    """4-bit integer quantized ESPCN model for BSD300 using common
    integer weight quantization with per-channel scales"""
    return quant_espcn_v1(1, 3, 4, 4, CommonIntWeightPerChannelQuant)


def quant_espcn_x3_v2_4b():
    """4-bit integer quantized ESPCN model for BSD300 using integer
    weight normalization-based quantizer with L2-norm"""
    return quant_espcn_v1(1, 3, 4, 4, Int8WeightNormL2PerChannelFixedPoint) 


def quant_espcn_x3_v3_4b():
    """4-bit integer quantized ESPCN model for BSD300 using integer
    accumulator-aware weight quantizer"""
    return quant_espcn_v1(1, 3, 4, 4, Int8AccumulatorAwareWeightQuant)
