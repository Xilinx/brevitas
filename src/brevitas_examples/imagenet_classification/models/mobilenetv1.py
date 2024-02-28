"""
Modified from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models

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

__all__ = ['quant_mobilenet_v1']

from torch import nn
from torch.nn import Sequential

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.nn import TruncAvgPool2d
from brevitas.quant import Int32Bias

from .common import CommonIntWeightPerChannelQuant
from .common import CommonIntWeightPerTensorQuant
from .common import CommonUintActQuant


class DwsConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            act_bit_width,
            weight_bit_width,
            weight_quant=CommonIntWeightPerChannelQuant,
            pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            act_bit_width=act_bit_width)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            act_bit_width=act_bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            weight_quant=CommonIntWeightPerChannelQuant,
            activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=weight_quant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_stats_permute_dims=(1, 0, 2, 3),
            scaling_per_output_channel=activation_scaling_per_channel,
            return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MobileNet(nn.Module):

    def __init__(
            self,
            channels,
            first_stage_stride,
            act_bit_width,
            weight_bit_width,
            round_average_pool=True,
            weight_quant=CommonIntWeightPerChannelQuant,
            first_layer_bit_width=8,
            first_layer_weight_quant=CommonIntWeightPerChannelQuant,
            last_layer_weight_quant=CommonIntWeightPerTensorQuant,
            last_layer_bit_width=8,
            avg_pool_kernel_size=7,
            first_layer_stride=2,
            in_channels=3,
            num_classes=1000):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]

        self.features = Sequential()
        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=first_layer_stride,
            weight_bit_width=first_layer_bit_width,
            weight_quant=first_layer_weight_quant,
            act_bit_width=act_bit_width,
            activation_scaling_per_channel=True)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    act_bit_width=act_bit_width,
                    weight_bit_width=weight_bit_width,
                    weight_quant=weight_quant,
                    pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        # Exporting to torch or ONNX qcdq requires round
        avgpool_float_to_int_impl_type = 'ROUND' if round_average_pool else 'FLOOR'
        self.final_pool = TruncAvgPool2d(
            kernel_size=avg_pool_kernel_size,
            stride=1,
            bit_width=last_layer_bit_width,
            float_to_int_impl_type=avgpool_float_to_int_impl_type)
        self.output = QuantLinear(
            in_channels,
            num_classes,
            bias=True,
            bias_quant=Int32Bias,
            weight_quant=last_layer_weight_quant,
            weight_bit_width=last_layer_bit_width)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


def quant_mobilenet_v1(cfg):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')
    round_avgpool = cfg.getboolean('QUANT', 'ROUND_AVG_POOL')

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        round_average_pool=round_avgpool,
        act_bit_width=bit_width,
        weight_bit_width=bit_width,
        last_layer_bit_width=bit_width)

    return net
