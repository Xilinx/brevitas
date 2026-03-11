# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Tuple

import torch


def mantissa_bits_to_float(bits: str, frexp_compatible: bool = False, normal: bool = True) -> float:
    # computes the decimal place value from a given binary tensor
    res = 1.0 if normal else 0.0
    for i, val in enumerate(bits):
        # iterating through from left to right
        res += ((2 ** -(i + 1)) * float(val))
    if frexp_compatible:
        return res / 2.
    else:
        return res


def get_minifloat_value(exponent: str, mantissa: str, exponent_bias: int) -> float:
    """
    Returns the minifloat value for a given exponent, mantissa and exponent_bias.
    It expects the exponent and mantissa in their binary format.
    """
    exponent_value = int(exponent, 2)

    if exponent_value == 0:  # subnormal
        exponent_bias -= 1  # exponent is e_min
        mantissa_value = mantissa_bits_to_float(mantissa, normal=False)
    else:  # normal
        mantissa_value = mantissa_bits_to_float(mantissa, normal=True)

    return (2 ** (exponent_value - exponent_bias)) * mantissa_value


def get_max_available_float(
        exponent_bit_width: int,
        mantissa_bit_width: int,
        exponent_bias: int,
        nan_values: Tuple[str],
        inf_values: Tuple[str],
        saturating: bool) -> torch.Tensor:
    # Idea: take the smallest NaN/inf value, set max_value to the next smaller one
    # inf without NaN not possible
    if inf_values is None and nan_values is None:
        # saturating has to be True if no NaN/inf value are used
        assert saturating, 'cannot be non-saturating without NaN/inf values'
        # no special cases, max_value is using all bits for exponent and mantissa
        exponent = '1' * exponent_bit_width
        mantissa = '1' * mantissa_bit_width
    elif nan_values is not None:
        # we at least have values for NaN, so initiate MaxValInfNaN
        special_values = nan_values + inf_values if inf_values is not None else nan_values

        # check that NaN/inf values are all mantissa_bit_width long
        if any(map(lambda x: len(x) > mantissa_bit_width, special_values)):
            raise RuntimeError('NaN/inf codes need to be the same length as the mantissa.')

        # get the minimum special case, our max value is the next smaller value
        min_special_case = min(map(lambda x: int(x, 2), special_values))

        max_value_mantissa = min_special_case - 1

        if max_value_mantissa < 0:
            # all mantissa values are used, so we need to use decrease exponent values
            exponent = '1' * (exponent_bit_width - 1)
            # add trailing 0 to reach bit width
            exponent += '0'
            # since we decreased exponent, we can use full mantissa
            mantissa = '1' * mantissa_bit_width
        else:
            # there is a free mantissa code, so use full exponent
            exponent = '1' * exponent_bit_width
            # get binary code for max_value_mantissa in the number of mantissa bits
            mantissa = format(max_value_mantissa, f'0{mantissa_bit_width}b')
    else:
        # no NaN values but inf values
        raise RuntimeError('Minifloat Error: inf value cannot exist without NaN value.')

    # we don't need the sign since we're looking for the max value
    max_value = get_minifloat_value(
        exponent=exponent, mantissa=mantissa, exponent_bias=exponent_bias)
    return max_value


def get_min_available_float(
        exponent_bit_width: int, mantissa_bit_width: int, exponent_bias: int) -> torch.Tensor:
    """
    Returns the minimum subnormal minifloat value for a given exponent and mantissa
    bit-width, and exponent bias.
    """
    exponent = '0' * exponent_bit_width
    mantissa = '0' * (mantissa_bit_width - 1) + '1'

    min_value = get_minifloat_value(
        exponent=exponent, mantissa=mantissa, exponent_bias=exponent_bias)
    return min_value


# TODO: Allow dynamically changing this value at runtime
def get_midmax_mantissa_bit_bias(
        mantissa_bit_width: int, nan_values: Tuple[str], inf_values: Tuple[str]) -> float:
    # Calculate how much bias needs to be added midmax calculation, based on the amount of reserved values for inf, nan
    num_inf_values = 0 if inf_values is None else len(inf_values)
    num_nan_values = 0 if nan_values is None else len(nan_values)
    total_reserved_values = num_inf_values + num_nan_values
    excess_reserved_values = total_reserved_values % 2 ** mantissa_bit_width  # How many extra values are reserved for the highest valid exponent
    if excess_reserved_values == 0:
        return 0.0  # No special reserved mantissa values at maximum valid mantissa
    elif (excess_reserved_values + 1) == 2 ** mantissa_bit_width:
        return 0.0  # Edge case when only f'0{mantissa_bit_width}b' is representable at the maximum mantissa
    else:
        return torch.log2(torch.tensor(excess_reserved_values + 1)).item(
        )  # The number of bits of the mantissa that are consumed by the reserved values
