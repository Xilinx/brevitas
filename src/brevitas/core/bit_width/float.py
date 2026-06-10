# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch

from brevitas.core.utils import StatelessBuffer
from brevitas.function import compute_max_mantissa


class StaticMaxMantissa(torch.nn.Module):
    """
    Module that returns a maximum mantissa value computed once at initialization.

    Args:
        bit_width: the number of mantissa bits used to compute the maximum mantissa value.
        max_mantissa_round_impl (torch.nn.Module, optional): Module used to round the integer max
            mantissa value during the computation. Defaults to None, in which case
            compute_max_mantissa falls back to its previous closed-form implementation without
            applying any rounding function.
        device: Device on which to create the tensor. Default: None.
        dtype: Data type of the tensor. Default: None.

    Examples:
        >>> static_max = StaticMaxMantissa(3)
        >>> static_max(torch.randn(2))
        tensor(1.8750)

    Note:
        The maximum mantissa value is computed once during initialization and stored using
        StatelessBuffer, meaning it won't be saved as part of a checkpoint but will be properly
        handled during device transfers and dtype conversions. The rounding function used by
        compute_max_mantissa can be customized through dependency injection via
        max_mantissa_round_impl.
    """

    def __init__(
            self,
            bit_width,
            max_mantissa_round_impl: Optional[torch.nn.Module] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None):
        super().__init__()
        max_mantissa = compute_max_mantissa(
            torch.tensor(float(bit_width), device=device, dtype=dtype), max_mantissa_round_impl)
        self.compute_max_mantissa = StatelessBuffer(max_mantissa)

    def forward(self, x):
        return self.compute_max_mantissa()


class ComputeMaxMantissa(torch.nn.Module):
    """
    Module that computes the maximum mantissa value dynamically from input tensor.

    Args:
        max_mantissa_round_impl (torch.nn.Module, optional): Module used to round the integer max
            mantissa value during the computation. Defaults to None, in which case
            compute_max_mantissa falls back to its previous closed-form implementation without
            applying any rounding function.

    Examples:
        >>> compute_max = ComputeMaxMantissa()
        >>> input_tensor = torch.randn(2, 3)
        >>> max_mantissa = compute_max(input_tensor)

    Note:
        This module computes the maximum mantissa on-the-fly using the compute_max_mantissa
        function from brevitas.function. The rounding function used by compute_max_mantissa can
        be customized through dependency injection via max_mantissa_round_impl.
    """

    def __init__(self, max_mantissa_round_impl: Optional[torch.nn.Module] = None):
        super().__init__()
        self.max_mantissa_round_impl = max_mantissa_round_impl

    def forward(self, x):
        x = compute_max_mantissa(x, self.max_mantissa_round_impl)
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

    def __init__(
            self, exponent_bias: float, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))

    def forward(self):
        return self.exponent_bias()


class ComputeExponentBias(torch.nn.Module):
    """
    Module that returns a runtime-computed exponent bias value.

    Args:
        exponent_bit_width_impl: Module that returns the exponent bit width

    Examples:
        >>> exp_bias = ComputeExponentBias(4.)
        >>> exp_bias()
        tensor(7.)
    """

    def __init__(self, exponent_bit_width_impl: torch.nn.Module):
        super().__init__()
        self.exponent_bit_width_impl = exponent_bit_width_impl

    def forward(self):
        return 2 ** (self.exponent_bit_width_impl() - 1) - 1
