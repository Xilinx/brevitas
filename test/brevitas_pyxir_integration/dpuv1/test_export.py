import os

import torch
from torch import nn
from dependencies import Injector
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d, QuantEltwiseAdd
from brevitas.onnx import export_dpuv1_onnx
from brevitas.quant_tensor import QuantTensor

KERNEL_SIZE = 3
CHANNELS = 5
IN_SIZE = (1, CHANNELS, 10, 10)
FC_IN_SIZE = 80


class DPUv1WeightQuantInjector(Injector):
    quant_type = 'INT'
    bit_width = 8
    bit_width_impl_type = 'CONST'
    restrict_scaling_type = 'POWER_OF_TWO'
    scaling_per_output_channel = False
    scaling_impl_type = 'STATS'
    scaling_stats_op = 'MAX'
    narrow_range = True
    signed = True


class DPUv1BiasQuantInjector(Injector):
    quant_type = 'INT'
    bit_width = 8
    narrow_range = True
    signed = True


class DPUv1OutputQuantInjector(Injector):
    quant_type = 'INT'
    bit_width = 8
    bit_width_impl_type = 'CONST'
    restrict_scaling_type = 'POWER_OF_TWO'
    scaling_per_output_channel = False
    scaling_impl_type = 'CONST'
    min_val = - 1.0
    max_val = 1.0
    signed = True
    narrow_range = False


class DPUv1ActQuantInjector(Injector):
    quant_type = 'INT'
    bit_width = 8
    bit_width_impl_type = 'CONST'
    restrict_scaling_type = 'POWER_OF_TWO'
    scaling_per_output_channel = False
    scaling_impl_type = 'CONST'
    min_val = - 6.0
    max_val = 6.0
    signed = True
    narrow_range = False


class QuantModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = QuantConv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            weight_quant=DPUv1WeightQuantInjector,
            bias_quant=None,
            output_quant=DPUv1OutputQuantInjector,
            bias=False,
            return_quant_tensor=True)
        self.act1 = QuantReLU(
            act_quant=DPUv1ActQuantInjector,
            return_quant_tensor=True)
        self.conv2 = QuantConv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            weight_quant=DPUv1WeightQuantInjector,
            bias_quant=None,
            output_quant=DPUv1OutputQuantInjector,
            bias=False,
            return_quant_tensor=True)
        self.act2 = QuantReLU(
            act_quant=DPUv1ActQuantInjector,
            return_quant_tensor=True)
        self.conv3 = QuantConv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            weight_quant=DPUv1WeightQuantInjector,
            bias_quant=None,
            output_quant=DPUv1OutputQuantInjector,
            bias=False,
            return_quant_tensor=True)
        self.act3 = QuantReLU(
            act_quant=DPUv1ActQuantInjector,
            return_quant_tensor=False)
        self.max_pool = QuantMaxPool2d(
            kernel_size=KERNEL_SIZE,
            stride=1,
            return_quant_tensor=True)
        self.eltwise_add = QuantEltwiseAdd(
            input_quant=DPUv1OutputQuantInjector,
            output_quant=DPUv1ActQuantInjector,
            return_quant_tensor=True)
        self.linear = nn.Linear(FC_IN_SIZE, CHANNELS)

    def forward(self, x):
        x = self.conv1(x)
        act1 = self.act1(x)
        x = self.conv2(act1)
        x = self.act2(x)
        max_pool = self.max_pool(act1)
        x = self.eltwise_add(x, max_pool)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(int(x.size(0)), -1)
        x = self.linear(x)
        return x


def test_export():
    x = QuantTensor(
        torch.randn(IN_SIZE),
        scale=torch.tensor(2.0 ** (-7)),
        bit_width=torch.tensor(8),
        signed=True)
    mod = QuantModel()
    # Export quantized model to ONNX
    export_dpuv1_onnx(mod, input_shape=IN_SIZE, input_t=x, export_path='quant_model.onnx',
                      input_names=["input_%d" % i for i in range(5)],
                      output_names=["output"])