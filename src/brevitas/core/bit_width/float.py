# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch

from brevitas.core.utils import StatelessBuffer
from brevitas.function import compute_max_mantissa


class StaticMaxMantissa(torch.nn.Module):
    """
    Module that returns a pre-computed maximum mantissa value.

    Args:
        compute_max_mantissa (torch.Tensor): Pre-computed maximum mantissa tensor.

    Examples:
        >>> max_mantissa = torch.tensor(7.0)
        >>> static_max = StaticMaxMantissa(max_mantissa)
        >>> static_max(torch.randn(2))
        tensor(7.)

    Note:
        The pre-computed mantissa value is stored using StatelessBuffer, meaning it won't be saved as part of
        a checkpoint but will be properly handled during device transfers and dtype conversions.
    """

    def __init__(
            self,
            compute_max_mantissa: torch.Tensor,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.compute_max_mantissa = StatelessBuffer(
            torch.tensor(compute_max_mantissa, device=device, dtype=dtype))

    def forward(self, x):
        return self.compute_max_mantissa()


class ComputeMaxMantissa(torch.nn.Module):
    """
    Module that computes the maximum mantissa value dynamically from input tensor.

    Examples:
        >>> compute_max = ComputeMaxMantissa()
        >>> input_tensor = torch.randn(2, 3)
        >>> max_mantissa = compute_max(input_tensor)

    Note:
        This module computes the maximum mantissa on-the-fly using the compute_max_mantissa
        function from brevitas.function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = compute_max_mantissa(x)
        return x


class StaticExponentBias(torch.nn.Module):
    """
    Module that returns a constant exponent bias value.

    Args:
        exponent_bias: Exponent bias value to be converted to float.
        device: Device on which to create the tensor. Default: None.
        dtype: Data type of the tensor. Default: None.

    Examples:
        >>> exp_bias = StaticExponentBias(127)
        >>> exp_bias()
        tensor(127.)

    Note:
        The exponent bias is stored using StatelessBuffer, meaning it won't be saved as part of
        a checkpoint but will be properly handled during device transfers and dtype conversions.
    """

    def __init__(self, exponent_bias, device=None, dtype=None):
        super().__init__()
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))

    def forward(self):
        return self.exponent_bias()
