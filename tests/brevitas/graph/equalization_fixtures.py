# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn
from torchvision import models

from brevitas import torch_version
from brevitas.graph.equalize import _cross_layer_equalization
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Act
from brevitas.quant.experimental.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

SEED = 123456
ATOL = 1e-3

MODELS = {
    'vit_b_32': [0.396, 0.657],
    'shufflenet_v2_x0_5': [0.318, 0.649],
    'mobilenet_v2': [0.161, 0.320],
    'resnet18': [0.487, 0.952],
    'googlenet': [0.495, 0.982],
    'inception_v3': [0.497, 0.989],
    'alexnet': [0.875, 0.875],}

IN_SIZE_CONV = (1, 3, 224, 224)
IN_SIZE_LINEAR = (1, 224, 3)
IN_SIZE_CONV_SMALL = (1, 3, 32, 32)


def equalize_test(regions, merge_bias, bias_shrinkage, scale_computation_type):
    scale_factors_regions = []
    for i in range(3):
        for region in regions:
            scale_factors_region = _cross_layer_equalization(
                region,
                merge_bias=merge_bias,
                bias_shrinkage=bias_shrinkage,
                scale_computation_type=scale_computation_type)
            if i == 0:
                scale_factors_regions.append(scale_factors_region)
    return scale_factors_regions


@pytest_cases.fixture
@pytest_cases.parametrize(
    "model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()],
    ids=[model_name for model_name, _ in MODELS.items()])
def model_coverage(model_dict: dict):
    model_name, coverage = model_dict

    if model_name == 'googlenet' and torch_version == version.parse('1.8.1'):
        pytest.skip(
            'Skip because of PyTorch error = AttributeError: \'function\' object has no attribute \'GoogLeNetOutputs\' '
        )
    if 'vit' in model_name and torch_version < version.parse('1.13'):
        pytest.skip(
            f'ViT supported from torch version 1.13, current torch version is {torch_version}')

    kwargs = dict()
    if model_name in ('inception_v3', 'googlenet'):
        kwargs['transform_input'] = False
    model = getattr(models, model_name)(pretrained=True, **kwargs)

    return model, coverage


@pytest_cases.fixture
def bnconv_model():

    class BNConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            # Simulate statistics gathering
            self.bn.running_mean.data = torch.randn_like(self.bn.running_mean)
            self.bn.running_var.data = torch.abs(torch.randn_like(self.bn.running_var))
            # Simulate learned parameters
            self.bn.weight.data = torch.randn_like(self.bn.weight)
            self.bn.bias.data = torch.randn_like(self.bn.bias)
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv(x)
            return x

    return BNConvModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def linearmha_model(bias, add_bias_kv, batch_first):
    if torch_version < version.parse('1.9.1'):
        pytest.skip(f"batch_first not supported in MHA with torch version {torch_version}")

    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class LinearMhaModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 24)
            self.mha = nn.MultiheadAttention(
                24, 3, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            x, _ = self.mha(x, x, x)
            return x

    return LinearMhaModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def layernormmha_model(bias, add_bias_kv, batch_first):
    if torch_version < version.parse('1.9.1'):
        pytest.skip(f"batch_first not supported in MHA with torch version {torch_version}")

    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class LayerNormMhaModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(3)
            # Simulate learned parameters
            self.layernorm.weight.data = torch.randn_like(self.layernorm.weight)
            self.layernorm.bias.data = torch.randn_like(self.layernorm.bias)
            self.mha = nn.MultiheadAttention(
                3, 3, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.layernorm(x)
            x = self.relu(x)
            x, _ = self.mha(x, x, x)
            return x

    return LayerNormMhaModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def mhalinear_model(bias, add_bias_kv, batch_first):
    if torch_version < version.parse('1.9.1'):
        pytest.skip(f"batch_first not supported in MHA with torch version {torch_version}")

    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class MhaLinearModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(
                3, 1, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.linear = nn.Linear(3, 6)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.mha(x, x, x)
            x = self.relu(x)
            x = self.linear(x)
            return x

    return MhaLinearModel


@pytest_cases.fixture
def convdepthconv_model():

    class ConvDepthConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.conv_0 = nn.Conv2d(16, 16, kernel_size=1, groups=16)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            return x

    return ConvDepthConvModel


@pytest_cases.fixture
def convbn_model():

    class ConvBNModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 128, kernel_size=3)
            self.bn = nn.BatchNorm2d(128)
            # Simulate statistics gathering
            self.bn.running_mean.data = torch.randn_like(self.bn.running_mean)
            self.bn.running_var.data = torch.abs(torch.randn_like(self.bn.running_var))
            # Simulate learned parameters
            self.bn.weight.data = torch.randn_like(self.bn.weight)
            self.bn.bias.data = torch.randn_like(self.bn.bias)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.bn(x)
            return x

    return ConvBNModel


@pytest_cases.fixture
def residual_model():

    class ResidualModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1)
            self.conv_0 = nn.Conv2d(16, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            start = x
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = start + x
            return x

    return ResidualModel


@pytest_cases.fixture
def srcsinkconflict_model():
    """
    In this example, conv_0 is both a src and sink.
    """

    class ResidualSrcsAndSinkModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv_start = nn.Conv2d(3, 3, kernel_size=1)
            self.conv = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_0 = nn.Conv2d(3, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            start = self.conv_start(x)
            x = self.conv_0(start)
            x = start + x
            x = self.relu(x)
            x = self.conv(x)
            return x

    return ResidualSrcsAndSinkModel


@pytest_cases.fixture
def mul_model():

    class ResidualSrcsAndSinkModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv_1 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_0 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_end = nn.Conv2d(3, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x_0 = self.conv_0(x)
            x_1 = self.conv_1(x)
            x = x_0 * x_1
            x = self.relu(x)
            x = self.conv_end(x)
            return x

    return ResidualSrcsAndSinkModel


@pytest_cases.fixture
def convgroupconv_model():

    class ConvGroupConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.conv_0 = nn.Conv2d(16, 32, kernel_size=1, groups=2)
            self.conv_1 = nn.Conv2d(32, 64, kernel_size=1, groups=4)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return ConvGroupConvModel


@pytest_cases.fixture
def convtranspose_model():

    class ConvTransposeModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU()
            self.conv_0 = nn.ConvTranspose2d(in_channels=3, out_channels=8, kernel_size=3)
            self.conv_1 = nn.ConvTranspose2d(in_channels=8, out_channels=32, kernel_size=3)

        def forward(self, x):
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return ConvTransposeModel


list_of_fixtures = [
    'residual_model',
    'srcsinkconflict_model',
    'mul_model',
    'bnconv_model',
    'convdepthconv_model',
    'linearmha_model',
    'mhalinear_model',
    'layernormmha_model',
    'convgroupconv_model',
    'convtranspose_model']

toy_model = fixture_union('toy_model', list_of_fixtures, ids=list_of_fixtures)

RESNET_18_REGIONS = [
    [('layer3.0.bn1',), ('layer3.0.conv2',)],
    [('layer4.1.bn1',), ('layer4.1.conv2',)],
    [('layer2.1.bn1',), ('layer2.1.conv2',)],
    [('layer3.1.bn1',), ('layer3.1.conv2',)],
    [('layer1.0.bn1',), ('layer1.0.conv2',)],
    [('layer3.0.bn2', 'layer3.0.downsample.1', 'layer3.1.bn2'),
     ('layer3.1.conv1', 'layer4.0.conv1', 'layer4.0.downsample.0')],
    [('layer4.0.bn1',), ('layer4.0.conv2',)],
    [('layer2.0.bn2', 'layer2.0.downsample.1', 'layer2.1.bn2'),
     ('layer2.1.conv1', 'layer3.0.conv1', 'layer3.0.downsample.0')],
    [('layer1.1.bn1',), ('layer1.1.conv2',)],
    [('bn1', 'layer1.0.bn2', 'layer1.1.bn2'),
     ('layer1.0.conv1', 'layer1.1.conv1', 'layer2.0.conv1', 'layer2.0.downsample.0')],
    [('layer2.0.bn1',), ('layer2.0.conv2',)],
    [('layer4.0.bn2', 'layer4.0.downsample.1', 'layer4.1.bn2'), ('fc', 'layer4.1.conv1')],]


input_quant, weight_quant = pytest_cases.param_fixtures("input_quant, weight_quant", [(None, Int8WeightPerTensorFloat), (Int8ActPerTensorFloat, Int8WeightPerTensorFloat), (MXInt8Act, MXInt8Weight), (MXFloat8e4m3Act, MXFloat8e4m3Weight)])


@pytest_cases.fixture
def quant_conv_with_input_quant_model(input_quant, weight_quant):

    class QuantConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv_0 = qnn.QuantConv2d(
                3, 16, kernel_size=3)  # gpxq tests assume no quant on first layer
            self.conv_1 = qnn.QuantConv2d(
                16, 32, kernel_size=3, input_quant=input_quant, weight_quant=weight_quant)

        def forward(self, x):
            x = self.conv_0(x)
            x = torch.relu(x)
            x = self.conv_1(x)
            return x

    return QuantConvModel


@pytest_cases.fixture
def quant_convdepthconv_model():

    class QuantConvDepthConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = qnn.QuantConv2d(3, 16, kernel_size=3)
            self.conv_0 = qnn.QuantConv2d(16, 16, kernel_size=1, groups=16)
            self.relu = qnn.QuantReLU(return_quant_tensor=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            return x

    return QuantConvDepthConvModel


@pytest_cases.fixture
def quant_residual_model(input_quant, weight_quant):

    class QuantResidualModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = qnn.QuantConv2d(
                3, 16, kernel_size=1, input_quant=input_quant, weight_quant=weight_quant)
            self.conv_0 = qnn.QuantConv2d(
                16, 3, kernel_size=1, input_quant=input_quant, weight_quant=weight_quant)
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)

        def forward(self, x):
            start = x
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = start + x

            return x

    return QuantResidualModel


@pytest_cases.fixture
def quant_convtranspose_model(input_quant, weight_quant):

    class QuantConvTransposeModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)
            self.conv_0 = qnn.QuantConvTranspose2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.conv_1 = qnn.QuantConvTranspose2d(
                in_channels=8,
                out_channels=32,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)

        def forward(self, x):
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return QuantConvTransposeModel


list_of_quant_fixtures = [
    'quant_conv_with_input_quant_model',
    'quant_convdepthconv_model',
    'quant_residual_model',
    'quant_convtranspose_model']

toy_quant_model = fixture_union(
    'toy_quant_model', list_of_quant_fixtures, ids=list_of_quant_fixtures)

## List of Rotation fixtures


@pytest_cases.fixture
def linear_rms():

    class LinearRMSModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            self.linear.weight.data.fill_(2.)
            self.linear.bias.data.fill_(1.)
            self.rms = nn.RMSNorm(4)
            self.rms.weight.data = torch.randn_like(
                self.rms.weight.data)  # Change learned parameters
            self.linear_1 = nn.Linear(4, 8, bias=False)
            self.linear_1.weight.data.fill_(2.)
            self.linear_2 = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            x = self.linear(x)
            x = self.rms(x)
            x = self.linear_1(x)
            x = self.linear_2(x) * x
            x = torch.matmul(x.flatten(1), x.flatten(1).t())

            return x

    return LinearRMSModel


list_of_rotation_mixtures = ['linear_rms']

rotation_fixtures = fixture_union(
    'rotation_fixtures', list_of_rotation_mixtures, ids=list_of_rotation_mixtures)
