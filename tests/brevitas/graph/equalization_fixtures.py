# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn

from brevitas import torch_version

MODELS = {
    'vit_b_32': [0.396, 0.396],
    'shufflenet_v2_x0_5': [0.8141, 0.8230],
    'mobilenet_v2': [0.6571, 0.6571],
    'resnet18': [0.9756, 0.9756],
    'googlenet': [0.4956, 0.4956],
    'inception_v3': [0.4973, 0.4973],
    'alexnet': [0.875, 0.875],}


IN_SIZE_CONV = (1, 3, 224, 224)
IN_SIZE_LINEAR = (1, 224, 3)

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

        def forward(self, x):
            x = self.bn(x)
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
    class LinearMhaModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3,24)
            self.mha = nn.MultiheadAttention(24,3,0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
        def forward(self, x):
            x = self.linear(x)
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
    class LayerNormMhaModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(3)
            # Simulate learned parameters
            self.layernorm.weight.data = torch.randn_like(self.layernorm.weight)
            self.layernorm.bias.data = torch.randn_like(self.layernorm.bias)
            self.mha = nn.MultiheadAttention(3,3,0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
        def forward(self, x):
            x = self.layernorm(x)
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
    class MhaLinearModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(3,1,0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.linear = nn.Linear(3,6)
        def forward(self, x):
            x, _ = self.mha(x, x, x)
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

        def forward(self, x):
            x = self.conv(x)
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

        def forward(self, x):
            x = self.conv(x)
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

        def forward(self, x):
            start = x
            x = self.conv(x)
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

        def forward(self, x):
            start = self.conv_start(x)
            x = self.conv_0(start)
            x = start + x
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

        def forward(self, x):
            x_0 = self.conv_0(x)
            x_1 = self.conv_1(x)
            x = x_0 * x_1
            x = self.conv_end(x)
            return x

    return ResidualSrcsAndSinkModel


list_of_fixtures = ['residual_model', 'srcsinkconflict_model', 'mul_model',
                    'convbn_model', 'bnconv_model', 'convdepthconv_model',
                    'linearmha_model', 'mhalinear_model', 'layernormmha_model']

toy_model = fixture_union('toy_model', list_of_fixtures, ids=list_of_fixtures)
