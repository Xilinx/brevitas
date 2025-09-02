# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.core.utils import StatelessBuffer
from brevitas.function import compute_max_mantissa


class StaticMaxMantissa(torch.nn.Module):

    def __init__(self, pre_computed_max_mantissa: torch.Tensor):
        super().__init__()
        self.pre_computed_max_mantissa = pre_computed_max_mantissa

    def forward(self, x):
        return self.pre_computed_max_mantissa


class ComputeMaxMantissa(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = compute_max_mantissa(x)
        return x


class StaticExponentBias(torch.nn.Module):

    def __init__(self, exponent_bias, device=None, dtype=None):
        super().__init__()
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))

    def forward(self):
        return self.exponent_bias()
