# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def mantissa_bits_to_float(bits: str, frexp_compatible: bool = False) -> float:
    # computes the decimal place value from a given binary tensor
    res = 1.0
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
    mantissa_value = mantissa_bits_to_float(mantissa)
    return 2 ** (exponent_value - exponent_bias) * mantissa_value
