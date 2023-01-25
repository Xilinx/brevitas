import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn

MODELS = [
    'shufflenet_v2_x0_5',
    'densenet121',
    'mobilenet_v2',
    'resnet18',
    'googlenet',
    'inception_v3',
    'alexnet',
]

@pytest_cases.fixture
def bnconv_model():
    class BNConvModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
        def forward(self, x):
            x = self.bn(x)
            x = self.conv(x)
            return x
    return BNConvModel

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
def cat_model():
    class CatModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.conv_0 = nn.Conv2d(3, 16, kernel_size=3)
            self.conv_1 = nn.Conv2d(32, 8, kernel_size=3)
        def forward(self, x):
            x_0 = self.conv(x)
            x_1 = self.conv_0(x)
            x = torch.cat([x_1, x_0], 1)
            x = self.conv_1(x)
            return x
    return CatModel


all_models = fixture_union('all_models', ['bnconv_model', 'convdepthconv_model', 'residual_model',
                                          'cat_model', 'srcsinkconflict_model'])
