# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor


def mantissa_bits_to_float(bits: Tensor, frexp_compatible: bool = False) -> float:
    # computes the decimal place value from a given binary tensor
    res = 1.0
    for i, val in enumerate(bits):
        # iterating through from left to right
        res += ((2 ** -(i + 1)) * val)
    if frexp_compatible:
        return res / 2.
    else:
        return res


def get_minifloat_value(exponent: Tensor, mantissa: Tensor, exponent_bias: Tensor) -> Tensor:
    """
    Returns the minifloat value for a given exponent, mantissa and exponent_bias.
    It expects the exponent and mantissa in their binary format.
    """
    exponent_value = bits_to_dec(exponent)
    mantissa_value = mantissa_bits_to_float(mantissa)
    return torch.exp2(exponent_value - exponent_bias) * mantissa_value


def dec_to_bits(value: Tensor, bits: int) -> Tensor:
    # set up mask
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(value.device, value.dtype)
    # add dimension, bitwise_and gets the bits needed for the value, the rest is converting to byte
    return value.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def bits_to_dec(bits: Tensor) -> Tensor:
    # get num of bits used
    num_bits = len(bits)
    # convert by summing decimal values of set bits
    return torch.sum((2 ** torch.arange(num_bits - 1, -1, -1)) * bits)
