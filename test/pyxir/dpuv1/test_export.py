import torch
from torch import nn
from dependencies import Injector

from brevitas.nn import QuantConv2d, QuantReLU
from brevitas.onnx import export_dpuv1_onnx
from brevitas.quant_tensor import QuantTensor

KERNEL_SIZE = 3
IN_CHANNELS = 10
OUT_CHANNELS = 20
IN_SIZE = (1, IN_CHANNELS, 50, 50)


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
    scaling_impl_type = 'STATS'
    scaling_stats_permute_dims = (1, 0, 2, 3)
    scaling_stats_op = 'MAX'
    signed = True
    narrow_range = False


def test_simple_export():
    x = QuantTensor(
        torch.randn(IN_SIZE),
        scale=torch.tensor(2.0 ** (-7)),
        bit_width=torch.tensor(8),
        signed=True)
    conv1 = QuantConv2d(
        kernel_size=KERNEL_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        weight_quant=DPUv1WeightQuantInjector,
        bias_quant=DPUv1BiasQuantInjector,
        output_quant=DPUv1OutputQuantInjector,
        return_quant_tensor=True)
    act1 = QuantReLU(
        act_quant=DPUv1OutputQuantInjector,
        return_quant_tensor=True)
    conv2 = QuantConv2d(
        kernel_size=KERNEL_SIZE,
        in_channels=OUT_CHANNELS,
        out_channels=OUT_CHANNELS,
        weight_quant=DPUv1WeightQuantInjector,
        bias_quant=DPUv1BiasQuantInjector,
        output_quant=DPUv1OutputQuantInjector,
        return_quant_tensor=True)
    act2 = QuantReLU(
        act_quant=DPUv1OutputQuantInjector,
        return_quant_tensor=True)
    mod = nn.Sequential(*[conv1, act1, conv2, act2])
    export_dpuv1_onnx(mod, input_shape=IN_SIZE, input_t=x, export_path='dpuv1_2layers.onnx')