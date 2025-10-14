# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule wrappers for various variants of clamping.
"""
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.utils import StatelessBuffer
from brevitas.function import tensor_clamp
from brevitas.function.ops import max_float


class TensorClamp(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops.tensor_clamp`.

    Examples:
        >>> tensor_clamp = TensorClamp()
        >>> min_val = torch.tensor(-2.0)
        >>> max_val = torch.tensor(2.0)
        >>> tensor_clamp(torch.tensor([-3.0, 3.0]), min_val, max_val)
        tensor([-2.,  2.])
    """

    def __init__(self) -> None:
        super(TensorClamp, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: Tensor, min_val: Tensor, max_val: Tensor):
        return tensor_clamp(x, min_val=min_val, max_val=max_val)


class ScalarClamp(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~torch.clamp`.

    Examples:
        >>> scalar_clamp = ScalarClamp(min_val=-2.0, max_val=2.0)
        >>> scalar_clamp(torch.tensor([-3.0, 3.0]))
        tensor([-2.,  2.])
    """

    __constants__ = ['min_val', 'max_val']

    def __init__(self, min_val, max_val) -> None:
        super(ScalarClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class ClampMin(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~torch.clamp_min`.

    Examples:
        >>> clamp_min = ClampMin(min_val=-2.0)
        >>> clamp_min(torch.tensor(-3.0))
        tensor(-2.)
    """

    __constants__ = ['min_val']

    def __init__(self, min_val: float) -> None:
        super(ClampMin, self).__init__()
        self.min_val = min_val

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        return x.clamp_min(self.min_val)


class FloatClamp(brevitas.jit.ScriptModule):
    """"
    ScriptModule for clamping minifloat formats to their inf/NaN implementations.

    Currently, inf/NaN codes have to be encoded through the mantissa.
    I.e. setting inf to 1101.111 (E4M3) is not a valid code.
    """

    __constants__ = ['saturating', 'inf_values', 'nan_values', 'signed']

    def __init__(
            self,
            tensor_clamp_impl: Module,
            signed: bool,
            inf_values: Optional[Tuple[str]] = None,
            nan_values: Optional[Tuple[str]] = None,
            max_available_float: Optional[Tensor] = None,
            saturating: bool = True,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None) -> None:
        super(FloatClamp, self).__init__()

        self.tensor_clamp_impl = tensor_clamp_impl
        self.saturating = saturating
        self.inf_values = inf_values
        self.nan_values = nan_values
        self.signed = signed

        if max_available_float:
            max_available_float = torch.tensor(max_available_float, device=device, dtype=dtype)
            self.max_available_float = StatelessBuffer(max_available_float)
        else:
            self.max_available_float = None

    def inf_nan_clamp(self, x, inf_mask, p_max_val_mask, n_max_val_mask):

        # if non-saturating, we need to map values greater than max_val to nan or inf
        if self.inf_values is not None:
            # we have inf values, so we set abs values > max_value to +- inf, and leave inf at inf
            x[p_max_val_mask] = torch.tensor(float('inf'))
            x[n_max_val_mask] = torch.tensor(float('-inf'))
        elif self.nan_values is not None:
            # no inf values, so we need to map them to NaN
            full_max_val_mask = torch.logical_or(p_max_val_mask, n_max_val_mask)
            x[full_max_val_mask] = torch.tensor(float('nan'))

            # we also map the inf values to NaN in this case
            x[inf_mask] = torch.tensor(float('nan'))
        else:
            raise RuntimeError(
                "Clamping is not saturating, but neither `inf_values` nor `nan_values` is specified"
            )
        return x

    def saturating_clamp(self, x, max_value, min_value):
        return self.tensor_clamp_impl(x, min_val=min_value, max_val=max_value)

    @brevitas.jit.script_method
    def forward(
            self,
            x: Tensor,
            exponent_bit_width: Tensor,
            pre_compute_max_mantissa: Tensor,
            exponent_bias: Tensor):

        max_value = max_float(exponent_bit_width, pre_compute_max_mantissa, exponent_bias)
        max_value = max_value if self.max_available_float is None else torch.min(
            max_value, self.max_available_float())
        min_value = torch.tensor(0.) if not self.signed else -max_value

        # Compute masks
        if not self.saturating:
            inf_mask = x.isinf()
            p_max_val_mask = x > max_value
            n_max_val_mask = x < min_value

        # first clamp everything to +- max_value, basically the saturating case
        x = self.saturating_clamp(x, max_value, min_value)

        if not self.saturating:
            x = self.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x, self.saturating, self.inf_values, self.nan_values
