# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import torch

import brevitas
from brevitas.core.utils import StatelessBuffer


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
            bit_width: Union[int, float],
            max_mantissa_round_impl: Optional[torch.nn.Module] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None):
        super().__init__()
        # Reuse ComputeMaxMantissa so that the formula (and the optional rounding) lives in a
        # single place. The value is computed once here, eagerly, and cached in a StatelessBuffer.
        max_mantissa = ComputeMaxMantissa(max_mantissa_round_impl)(
            torch.tensor(float(bit_width), device=device, dtype=dtype))
        self.compute_max_mantissa = StatelessBuffer(max_mantissa)

    def forward(self, x: torch.Tensor):
        return self.compute_max_mantissa()


class ComputeMaxMantissa(brevitas.jit.ScriptModule):
    """
    Module that computes the maximum mantissa value dynamically from the input mantissa bit width.

    Args:
        max_mantissa_round_impl (torch.nn.Module, optional): Module used to round the integer max
            mantissa value ``2 ** (mantissa_bit_width + 1) - 1`` before scaling, enabling support
            for continuous (fractional) mantissa bit-widths (e.g. via a straight-through estimator).
            Defaults to None, in which case the closed-form implementation
            ``2 * (1 - 2 ** (-mantissa_bit_width - 1))`` is used and no rounding is applied.

    Examples:
        >>> compute_max = ComputeMaxMantissa()
        >>> compute_max(torch.tensor(3.))
        tensor(1.8750)

    Note:
        The rounding implementation is held as a submodule (rather than passed to a free function),
        so that this module remains compatible with the TorchScript JIT: TorchScript supports
        calling submodules but does not support ``torch.nn.Module`` values as function arguments.
        When ``max_mantissa_round_impl`` is None the previous closed-form behaviour is preserved
        and no rounding implementation is required.
    """

    def __init__(self, max_mantissa_round_impl: Optional[torch.nn.Module] = None):
        super().__init__()
        self.max_mantissa_round_impl = max_mantissa_round_impl

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.max_mantissa_round_impl is not None:
            return self.max_mantissa_round_impl(torch.exp2(x + 1) - 1) * torch.exp2(-x)
        # No rounding implementation: fall back to the previous closed-form computation.
        return 2 * (1 - 2 ** (-x - 1))


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
