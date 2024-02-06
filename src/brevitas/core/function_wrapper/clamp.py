# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule wrappers for various variants of clamping.
"""
from typing import Tuple

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

    Currently, inf/NaN codes have to be encoded through the mantissa.
    I.e. setting inf to 1101.111 (E4M3) is not a valid code.
    """

    __constants__ = ['nan_values', 'inf_values', 'saturating']

    def __init__(self, nan_values: Tuple[str], inf_values: Tuple[str], saturating: bool) -> None:
        super(FloatClamp, self).__init__()
        # TODO: check that NaN/inf values are all mantissa_bit_width long
        self.nan_values = nan_values if nan_values is not None else tuple()
        self.inf_values = inf_values if inf_values is not None else tuple()
        # inf without NaN not possible
        if self.inf_values is not None and self.nan_values is None:
            raise RuntimeError('Minifloat Error: inf value cannot exist without NaN value.')
        self.saturating = saturating

    def mantissa_bits_to_float(self, bits: str, frexp_compatible: bool = False) -> float:
        res = 1.0
        for i, val in enumerate(bits):
            # iterating through from left to right
            res += ((2 ** -(i + 1)) * float(val))
        if frexp_compatible:
            return res / 2.
        else:
            return res

    def get_minifloat_value(
            self,
            exponent_string: str,
            mantissa_string: str,
            exponent_bias: Tensor,
            sign: str = '0') -> float:
        exponent_value = int(exponent_string, 2)
        mantissa_value = self.mantissa_bits_to_float(mantissa_string)
        return ((-1) ** float(sign)) * 2 ** (exponent_value - exponent_bias) * mantissa_value

    def get_max_value(
            self, exponent_bit_width: int, mantissa_bit_width: int, exponent_bias: Tensor) -> float:
        # calculate max possible value for this specific format
        if not self.nan_values and not self.inf_values:
            # we don't have any codes, so just return max possible value
            exponent_string = '1' * exponent_bit_width
            mantissa_string = '1' * mantissa_bit_width
        else:
            # idea: take inf and nan values, select the smallest, set max_value to smallest_val - 1
            min_special_case = min(map(lambda x: int(x, 2), self.nan_values + self.inf_values))
            max_value_mantissa = min_special_case - 1
            if max_value_mantissa < 0:
                # all mantissa values are used, so we need to use decrease exponent values
                exponent_string = '1' * (exponent_bit_width - 1)
                exponent_string += '0'  # add trailing 0 to reach bit width
                # since we decreased exponent, we can use full mantissa
                mantissa_string = '1' * mantissa_bit_width
            else:
                # there is a free mantissa code, so use full exponent
                exponent_string = '1' * exponent_bit_width
                # get binary code for max_value_mantissa in the number of mantissa bits
                mantissa_string = format(max_value_mantissa, f'0{mantissa_bit_width}b')

        # we don't need the sign since we're looking for the max value
        max_value = self.get_minifloat_value(
            exponent_string=exponent_string,
            mantissa_string=mantissa_string,
            exponent_bias=exponent_bias)
        return max_value

    @brevitas.jit.script_method
    def forward(
            self,
            x: Tensor,
            exponent_bit_width: Tensor,
            mantissa_bit_width: Tensor,
            exponent_bias: Tensor):
        max_value = self.get_max_value(
            exponent_bit_width=exponent_bit_width.int().item(),
            mantissa_bit_width=mantissa_bit_width.int().item(),
            exponent_bias=exponent_bias)
        # TODO: at this time, we just pass the codes for inf/NaN, we might need to change that
        return clamp_to_fp_encoding(
            x,
            exponent_bit_width,
            mantissa_bit_width,
            nan_values=self.nan_values,
            inf_values=self.inf_values,
            max_value=max_value,
            saturating=self.saturating)
