"""
Source: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv
MIT License
Copyright (c) 2019 Xilinx, Inc (Alessandro Pappalardo)
Copyright (c) 2018 Oleg Sémery
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__all__ = ['quant_proxylessnas_mobile14']

import torch.nn as nn

from brevitas.nn import QuantConv2d, QuantLinear, HadamardClassifier
from brevitas.nn import QuantAvgPool2d, QuantReLU, QuantIdentity
from brevitas.quant import IntBias

from .common import *


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight_bit_width,
            act_bit_width,
            act_scaling_per_channel,
            bias,
            groups=1,
            bn_eps=1e-5,
            shared_act=None,
            return_quant_tensor=False):
        super(ConvBlock, self).__init__()

        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=CommonIntWeightPerChannelQuant)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if shared_act is None:
            self.activ = QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                scaling_per_channel=act_scaling_per_channel,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                return_quant_tensor=return_quant_tensor)
        else:
            self.activ = shared_act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ProxylessBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bn_eps,
            expansion,
            bit_width,
            depthwise_bit_width,
            shared_act):
        super(ProxylessBlock, self).__init__()
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        if self.use_bc:
            self.bc_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bn_eps=bn_eps,
                act_scaling_per_channel=True,
                weight_bit_width=bit_width,
                bias=False,
                act_bit_width=depthwise_bit_width)

        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=mid_channels,
            bn_eps=bn_eps,
            act_scaling_per_channel=False,
            weight_bit_width=depthwise_bit_width,
            act_bit_width=bit_width,
            bias=False)
        self.pw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=bn_eps,
            weight_bit_width=bit_width,
            shared_act=shared_act,
            bias=False,
            act_bit_width=None,
            act_scaling_per_channel=None)

    def forward(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bn_eps,
            expansion,
            residual,
            shortcut,
            bit_width,
            depthwise_bit_width,
            shared_act):
        super(ProxylessUnit, self).__init__()
        assert residual or shortcut
        assert shared_act is not None
        self.residual = residual
        self.shortcut = shortcut

        if self.residual:
            self.body = ProxylessBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bn_eps=bn_eps,
                expansion=expansion,
                bit_width=bit_width,
                depthwise_bit_width=depthwise_bit_width,
                shared_act=shared_act)
            self.shared_act = shared_act

    def forward(self, x):
        if not self.residual:
            return x
        if not self.shortcut:
            x = self.body(x)
            return x
        identity = x
        x = self.body(x)
        x = identity + x
        x = self.shared_act(x)
        return x


class ProxylessNAS(nn.Module):
    def __init__(
            self,
            channels,
            init_block_channels,
            final_block_channels,
            residuals,
            shortcuts,
            kernel_sizes,
            expansions,
            bit_width,
            depthwise_bit_width,
            first_layer_weight_bit_width,
            hadamard_classifier,
            bn_eps=1e-3,
            in_channels=3,
            num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.features = nn.Sequential()

        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            bn_eps=bn_eps,
            act_scaling_per_channel=False,
            bias=False,
            act_bit_width=bit_width,
            weight_bit_width=first_layer_weight_bit_width)
        self.features.add_module("init_block", init_block)

        in_channels = init_block_channels
        shared_act = None

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            residuals_per_stage = residuals[i]
            shortcuts_per_stage = shortcuts[i]
            kernel_sizes_per_stage = kernel_sizes[i]
            expansions_per_stage = expansions[i]

            for j, out_channels in enumerate(channels_per_stage):
                residual = (residuals_per_stage[j] == 1)
                shortcut = (shortcuts_per_stage[j] == 1)
                kernel_size = kernel_sizes_per_stage[j]
                expansion = expansions_per_stage[j]
                stride = 2 if (j == 0) and (i != 0) else 1

                if not shortcut:
                    shared_act = QuantIdentity(
                        bit_width=bit_width, act_quant=CommonIntActQuant, return_quant_tensor=True)

                unit = ProxylessUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bn_eps=bn_eps,
                    expansion=expansion,
                    residual=residual,
                    shortcut=shortcut,
                    bit_width=bit_width,
                    depthwise_bit_width=depthwise_bit_width,
                    shared_act=shared_act)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels

            self.features.add_module("stage{}".format(i + 1), stage)

        final_block = ConvBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=bn_eps,
            act_scaling_per_channel=False,
            act_bit_width=bit_width,
            weight_bit_width=bit_width,
            bias=False,
            return_quant_tensor=True)
        self.features.add_module("final_block", final_block)
        in_channels = final_block_channels
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width)
        if hadamard_classifier:
            self.output = HadamardClassifier(
                in_channels=in_channels,
                out_channels=num_classes,
                fixed_scale=False)
        else:
            self.output = QuantLinear(
                in_features=in_channels,
                out_features=num_classes,
                bias=True,
                bias_quant=IntBias,
                weight_bit_width=bit_width,
                weight_quant=CommonIntWeightPerTensorQuant)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def quant_proxylessnas_mobile14(cfg):

    residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    channels = [[24], [40, 40, 40, 40], [56, 56, 56, 56], [112, 112, 112, 112, 136, 136, 136, 136],
                [256, 256, 256, 256, 448]]
    kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
    expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
    init_block_channels = 48
    final_block_channels = 1792
    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    bit_width = int(cfg.get('QUANT', 'BIT_WIDTH'))
    first_layer_weight_bit_width = int(cfg.get('QUANT', 'FIRST_LAYER_WEIGHT_BIT_WIDTH'))
    depthwise_bit_width = int(cfg.get('QUANT', 'DEPTHWISE_BIT_WIDTH'))
    hadamard_classifier = cfg.getboolean('MODEL', 'HADAMARD_CLASSIFIER')

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expansions=expansions,
        bit_width=bit_width,
        first_layer_weight_bit_width=first_layer_weight_bit_width,
        depthwise_bit_width=depthwise_bit_width,
        hadamard_classifier=hadamard_classifier)
    return net
