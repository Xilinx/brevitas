# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule wrappers for various variants of clamping.
"""

import torch
from torch import Tensor

import brevitas
from brevitas.function import clamp_to_fp_encoding
from brevitas.function import tensor_clamp


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
    """

    __constants__ = ['nan_value', 'inf_value', 'max_value', 'saturating']

    def __init__(self, nan_value: str, inf_value: str, max_value: float, saturating: bool) -> None:
        super(FloatClamp, self).__init__()
        self.nan_value = self.mantissa_bits_to_float(nan_value)
        self.inf_value = self.mantissa_bits_to_float(inf_value) if inf_value is not None else None
        self.max_value = max_value
        self.saturating = saturating

    def mantissa_bits_to_float(self, bits: str, frexp_compatible: bool = True) -> float:
        res = 1.0
        for i, val in enumerate(bits):
            # iterating through from left to right
            res += ((2 ** -(i + 1)) * float(val))
        if frexp_compatible:
            return res / 2.
        else:
            return res

    @brevitas.jit.script_method
    def forward(self, x: Tensor, exponent_bit_width: int, mantissa_bit_width: int):
        return clamp_to_fp_encoding(
            x,
            exponent_bit_width,
            mantissa_bit_width,
            nan_value=self.nan_value,
            inf_value=self.inf_value,
            max_value=self.max_value,
            saturating=self.saturating)
