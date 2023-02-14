import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn


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

toy_model = fixture_union('toy_model', [ 'residual_model', 'srcsinkconflict_model', 'mul_model',
                                        'convbn_model', 'bnconv_model'])
