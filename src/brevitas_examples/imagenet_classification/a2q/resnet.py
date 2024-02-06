# Copyright (c) 2024, Advanced Micro Devices, Inc.
# Copyright (c) 2017, liukuang
# All rights reserved.
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torch.nn.functional as F

from brevitas.nn.quant_layer import WeightQuantType
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas_examples.bnn_pynq.models.resnet import QuantBasicBlock
from brevitas_examples.bnn_pynq.models.resnet import QuantResNet


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain('relu'))
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)


class BasicBlock(nn.Module):
    """Basic block architecture modified for CIFAR10.
    Adapted from https://github.com/kuangliu/pytorch-cifar"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # using a convolution shortcut rather than identity
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ ResNet architecture modified for CIFAR10.
    Adapted from https://github.com/kuangliu/pytorch-cifar"""

    def __init__(self, block_impl, num_blocks, num_classes: int = 10):
        super(ResNet, self).__init__()

        # stride and padding of 1 with kernel size of 3, compared to ImageNet model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64
        self.layer1 = self.create_block(block_impl, 64, num_blocks[0], stride=1)
        self.layer2 = self.create_block(block_impl, 128, num_blocks[1], stride=2)
        self.layer3 = self.create_block(block_impl, 256, num_blocks[2], stride=2)
        self.layer4 = self.create_block(block_impl, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block_impl.expansion, num_classes)

        self.apply(weight_init)

    def create_block(self, block_impl, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_impl(self.in_planes, planes, stride))
            self.in_planes = planes * block_impl.expansion  # expand input planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def float_resnet18(num_classes: int = 10) -> ResNet:
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def quant_resnet18(
        num_classes: int = 10,
        act_bit_width: int = 8,
        acc_bit_width: int = 32,
        weight_bit_width: int = 8,
        weight_quant: WeightQuantType = Int8WeightPerChannelFloat) -> QuantResNet:
    weight_quant = weight_quant.let(accumulator_bit_width=acc_bit_width)
    model = QuantResNet(
        block_impl=QuantBasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        act_bit_width=act_bit_width,
        weight_bit_width=weight_bit_width,
        weight_quant=weight_quant,
        last_layer_weight_quant=Int8WeightPerChannelFloat,
        first_maxpool=False,
        zero_init_residual=False,
        round_average_pool=False)
    return model
