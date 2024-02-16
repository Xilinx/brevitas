# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule wrappers for various variants of clamping.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

import brevitas
from brevitas.core.utils import StatelessBuffer
from brevitas.function import max_float
from brevitas.function import tensor_clamp
from brevitas.utils.float_quant_utils import get_minifloat_value

P_INF_TENSOR = torch.tensor(float('inf'))
N_INF_TENSOR = torch.tensor(float('-inf'))
NAN_TENSOR = torch.tensor(float('nan'))


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

    __constants__ = [
        'exponent_bit_width',
        'mantissa_bit_width',
        'exponent_bias',
        'nan_values',
        'inf_values',
        'saturating']

    def __init__(
            self,
            exponent_bit_width: Tensor,
            mantissa_bit_width: Tensor,
            exponent_bias: Tensor,
            nan_values: Optional[Tuple[str]] = None,
            inf_values: Optional[Tuple[str]] = None,
            saturating: bool = False) -> None:
        super(FloatClamp, self).__init__()

        self.exponent_bit_width = exponent_bit_width
        self.mantissa_bit_width = mantissa_bit_width
        self.exponent_bias = exponent_bias

        self.nan_values = nan_values
        self.inf_values = inf_values
        self.saturating = saturating

        # inf without NaN not possible
        if self.inf_values is None and self.nan_values is None:
            self.max_val_impl = StatelessBuffer(
                max_float(
                    self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias()))
        elif self.nan_values is not None:
            # we at least have values for NaN, so initiate MaxValInfNaN
            self.max_val_impl = MaxFloatInfNaN(
                exponent_bit_width=self.exponent_bit_width,
                mantissa_bit_width=self.mantissa_bit_width,
                exponent_bias=self.exponent_bias,
                nan_values=self.nan_values,
                inf_values=self.inf_values,
                saturating=self.saturating)
        else:
            # no NaN values but inf values
            raise RuntimeError('Minifloat Error: inf value cannot exist without NaN value.')

        self.clamp_impl = CaseClamp(inf_values=self.inf_values, saturating=self.saturating)

    @brevitas.jit.script_method
    def forward(self, inp: Tensor):
        # get max value for the minifloat config
        max_value = self.max_val_impl()
        return self.clamp_impl(inp, max_value)


class MaxFloatInfNaN(brevitas.jit.ScriptModule):

    def __init__(
            self,
            exponent_bit_width: Tensor,
            mantissa_bit_width: Tensor,
            exponent_bias: Tensor,
            nan_values: Tuple[str],
            inf_values: Tuple[str],
            saturating: bool = False) -> None:
        super(MaxFloatInfNaN, self).__init__()
        self.exponent_bit_width = exponent_bit_width
        self.mantissa_bit_width = mantissa_bit_width
        self.exponent_bias = exponent_bias

        self.inf_values = inf_values
        self.nan_values = nan_values
        self.__special_values = nan_values + inf_values if inf_values is not None else nan_values

        self.saturating = saturating

        # check that NaN/inf values are all mantissa_bit_width long
        if any(map(lambda x: len(x) > mantissa_bit_width, self.__special_values)):
            raise RuntimeError('NaN/inf codes need to be the same length as the mantissa.')

        # move computation of min for forward pass here so it's jit compatible
        self.__min_special_case = min(map(lambda x: int(x, 2), self.__special_values))

    @brevitas.jit.script_method
    def forward(self):
        # idea: take inf and nan values, select the smallest, set max_value to smallest_val - 1
        max_value_mantissa = self.__min_special_case - 1

        if max_value_mantissa < 0:
            # all mantissa values are used, so we need to use decrease exponent values
            exponent_string = '1' * (self.exponent_bit_width - 1)
            exponent_string += '0'  # add trailing 0 to reach bit width
            # since we decreased exponent, we can use full mantissa
            mantissa_string = '1' * self.mantissa_bit_width
        else:
            # there is a free mantissa code, so use full exponent
            exponent_string = '1' * self.exponent_bit_width
            # get binary code for max_value_mantissa in the number of mantissa bits
            mantissa_string = format(max_value_mantissa, f'0{self.mantissa_bit_width}b')

        # we don't need the sign since we're looking for the max value
        max_value = get_minifloat_value(
            exponent_string=exponent_string,
            mantissa_string=mantissa_string,
            exponent_bias=self.exponent_bias)
        return max_value


class CaseClamp(brevitas.jit.ScriptModule):

    def __init__(self, inf_values: Tuple[str], saturating: bool) -> None:
        super(CaseClamp, self).__init__()
        self.inf_values = inf_values
        self.saturating = saturating

    @brevitas.jit.script_method
    def forward(self, x: Tensor, max_value: Tensor):
        # NaN values all stay at NaN, so no need to do anything with NaN values
        # get all positive inf values
        inf_mask = x.isinf()
        p_max_val_mask = x > max_value
        n_max_val_mask = -x > max_value

        if self.saturating:
            # clamp everything to +- max_value
            x = x.clamp(-max_value, max_value)
        else:
            if self.inf_values is not None:
                # we have inf values, so we set abs values > max_value to +- inf, and leave inf at inf
                x[p_max_val_mask] = P_INF_TENSOR
                x[n_max_val_mask] = N_INF_TENSOR
            else:
                # no inf values, so we need to map them to NaN
                full_max_val_mask = torch.logical_or(p_max_val_mask, n_max_val_mask)
                x[full_max_val_mask] = NAN_TENSOR

                # we also map the inf values to NaN in this case
                x[inf_mask] = NAN_TENSOR

        return x
