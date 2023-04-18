# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from brevitas.export import export_qonnx
import brevitas.nn as qnn
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor


def make_quant_conv2d(
        in_channels, out_channels, kernel_size, weight_bit_width, stride=1, padding=0, bias=False):
    return qnn.QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_bit_width=weight_bit_width,
        weight_scaling_per_output_channel=True)


class QuantBasicBlock(nn.Module):
    """
    Quantized BasicBlock implementation with extra relu activations to respect FINN constraints on the sign of residual
    adds. Ok to train from scratch, but doesn't lend itself to e.g. retrain from torchvision.
    """
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
            shared_quant_act=None,
            weight_bit_width=8,
            act_bit_width=8,
            bias=False):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = make_quant_conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.conv2 = make_quant_conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        # We add a ReLU activation here because FINN requires the same sign along residual adds
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if len(self.downsample):
            x = self.downsample(x)
        # Check that the addition is made explicitly among QuantTensor structures
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out = out + x
        out = self.relu_out(out)
        return out


class QuantResNet(nn.Module):

    def __init__(
            self,
            block_impl,
            num_blocks: List[int],
            first_maxpool=False,
            zero_init_residual=False,
            num_classes=10,
            weight_bit_width=8,
            act_bit_width=8):
        super(QuantResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = make_quant_conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, weight_bit_width=8)
        self.bn1 = nn.BatchNorm2d(64)
        shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act
        # MaxPool is typically present for ImageNet but not for CIFAR10
        if first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1, shared_quant_act = self._make_layer(
            block_impl, 64, num_blocks[0], 1, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer2, shared_quant_act = self._make_layer(
            block_impl, 128, num_blocks[1], 2, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer3, shared_quant_act = self._make_layer(
            block_impl, 256, num_blocks[2], 2, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer4, _ = self._make_layer(
            block_impl, 512, num_blocks[3], 2, shared_quant_act, weight_bit_width, act_bit_width)

        # Performs truncation to 8b (without rounding), which is supported in FINN
        self.final_pool = qnn.TruncAvgPool2d(kernel_size=4, trunc_quant=TruncTo8bit)
        # Keep last layer at 8b
        self.linear = qnn.QuantLinear(
            512 * block_impl.expansion, num_classes, weight_bit_width=8, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block_impl,
            planes,
            num_blocks,
            stride,
            shared_quant_act,
            weight_bit_width,
            act_bit_width):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = block_impl(
                self.in_planes, planes, stride, shared_quant_act, weight_bit_width, act_bit_width)
            layers.append(block)
            shared_quant_act = layers[-1].relu_out
            self.in_planes = planes * block_impl.expansion
        return nn.Sequential(*layers), shared_quant_act

    def forward(self, x: Tensor):
        # There is no input quantizer, we assume the input is already 8b RGB
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def quant_resnet18(cfg) -> QuantResNet:
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    model = QuantResNet(
        QuantBasicBlock, [2, 2, 2, 2],
        num_classes=num_classes,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width)
    return model
