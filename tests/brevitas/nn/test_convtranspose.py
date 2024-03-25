# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from brevitas.nn import QuantConvTranspose1d
from brevitas.nn import QuantConvTranspose2d
from brevitas.nn import QuantConvTranspose3d


def test_quantconvtranspose1d():
    in_channels = 16
    out_channels = 4
    kernel_size = 3

    input = torch.ones(10, in_channels, 50)

    normal = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2)
    normal_output = normal(input)

    quant = QuantConvTranspose1d(in_channels, out_channels, kernel_size, stride=2)

    # re-using weight and bias so the layers should give the same results
    quant.weight = normal.weight
    quant.bias = normal.bias
    quant_output = quant(input)

    assert torch.isclose(normal_output, quant_output, atol=0.01).all().item()


def test_quantconvtranspose2d():
    in_channels = 16
    out_channels = 4
    kernel_size = 3

    input = torch.ones(10, in_channels, 50, 100)

    normal = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)
    normal_output = normal(input)

    quant = QuantConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)

    # re-using weight and bias so the layers should give the same results
    quant.weight = normal.weight
    quant.bias = normal.bias
    quant_output = quant(input)

    assert torch.isclose(normal_output, quant_output, atol=0.01).all().item()


def test_quantconvtranspose3d():
    in_channels = 16
    out_channels = 4
    kernel_size = 3

    input = torch.ones(10, in_channels, 10, 50, 100)

    normal = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2)
    normal_output = normal(input)

    quant = QuantConvTranspose3d(in_channels, out_channels, kernel_size, stride=2)

    # re-using weight and bias so the layers should give the same results
    quant.weight = normal.weight
    quant.bias = normal.bias
    quant_output = quant(input)

    assert torch.isclose(normal_output, quant_output, atol=0.01).all().item()
