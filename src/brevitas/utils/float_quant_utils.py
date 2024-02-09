# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor


def mantissa_bits_to_float(bits: str, frexp_compatible: bool = False) -> float:
    res = 1.0
    for i, val in enumerate(bits):
        # iterating through from left to right
        res += ((2 ** -(i + 1)) * float(val))
    if frexp_compatible:
        return res / 2.
    else:
        return res


def get_minifloat_value(
        exponent_string: str,
        mantissa_string: str,
        exponent_bias: Tensor,
        sign: str = '0') -> float:
    exponent_value = int(exponent_string, 2)
    mantissa_value = mantissa_bits_to_float(mantissa_string)
    return ((-1) ** float(sign)) * 2 ** (exponent_value - exponent_bias) * mantissa_value
