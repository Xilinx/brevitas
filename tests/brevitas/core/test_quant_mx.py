# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Brief MXFP quantizer
"""

import struct
from typing import List, Optional, Tuple, Union

from hypothesis import given
import pytest_cases
import torch

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_linear import QuantLinear
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from tests.brevitas.hyp_helper import float_tensor_nz_st

torch.manual_seed(0)


# debug utility
def to_string(val: Union[torch.Tensor, float],
              spaced: bool = True,
              code: str = "f") -> Union[str, List[str]]:
    """ Debug util for visualizing float values """

    def scalar_to_string(val: float, spaced: bool) -> str:
        s = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!' + code, val))
        spaced = spaced and len(s) == 32
        return f"{s[0]} {s[1:9]} {s[9:]}" if spaced else s

    if isinstance(val, float):
        return scalar_to_string(val, spaced)
    val = val.view(-1)
    return [scalar_to_string(val[i].item(), spaced) for i in range(val.numel())]


# debug utility
def check_bits(val: Union[torch.Tensor, float], mbits: int) -> Tuple[bool, int]:
    """ return (too many precision bits, lowest mantissa bit) """
    strings = to_string(val, spaced=False)
    if isinstance(strings, str):
        strings = [strings]
    error, lowest = False, 0
    for s in strings:
        mant = s[9:]
        error = error or "1" in mant[mbits:]
        lowest = max(lowest, mant.find("1"))
    return error, lowest


# Avoid returning exp 0 if we is 0
def safe_frexp(x: torch.Tensor) -> torch.Tensor:
    """torch.frexp returns unbiased exponent 0 for 0.0, which is not what we want."""
    if x.is_cuda and x.dtype not in (torch.float32, torch.float16):
        x = x.float()  # no gpu support for frexp on bfloat16 or any float8
    return torch.where(x == 0.0, torch.tensor(-126, dtype=torch.int32), x.frexp().exponent - 1)


class MXFP:
    """
    MXFP - Quantize OCP MXFP floating point types.
    A type is defined as ebits, mbits, bias, and inf/nan handling.
    """
    CONFIG = dict(
        e5m2=(5, 2, 15, "ieee"),
        e4m3=(4, 3, 7, "fn"),
        e3m2=(3, 2, 3, "fnuz"),
        e2m3=(2, 3, 1, "fnuz"),
        e2m1=(2, 1, 1, "fnuz"))

    def __init__(self, name, tile_size: Optional[int] = 32):
        self.name = name.lower()
        assert self.name in self.CONFIG
        self.ebits, self.mbits, self.bias, self.infnan = self.CONFIG[self.name]
        self.tile_size = tile_size

    @property  # maximum unbiased exponent for this type
    def emax(self) -> int:
        return 2 ** self.ebits - 1 - self.bias - int(self.infnan == "ieee")

    @property  # minimum unbiased exponent for this type
    def emin(self) -> int:
        return 1 - self.bias

    @property  # maximum representable value; the "fn" reserves values for all non-sign bits == 1
    def maxval(self) -> float:
        return 2 ** self.emax * (2.0 - (1 + int(self.infnan == "fn")) * 2 ** (-self.mbits))

    @property  # for alternative scale selection
    def midmax(self) -> float:
        return (2 ** (self.emax + 1) - self.maxval) / 2. + self.maxval

    @property  # minimum representable positive value
    def minval(self) -> float:
        return 2 ** self.emin * 2 ** (-self.mbits)

    def quantize(self, tensor: torch.Tensor, axis: int = -1, select: bool = False):
        """
        Fake quantize along the indicated dimension. This method assumes the tile dimension is the size of the tile,
        so some reshaping and possibly padding is likely required.  From there, we have 5 needed lines of code.
        """
        exp = safe_frexp(tensor)  # safe_frexp pretends the mantissa is < 1.0
        shared = exp.amax(axis, keepdim=True)  # shared exponent per the OCP MX spec

        # This is an alternative to the OCP MX scale selection, which chooses the maximum exponent (maxexp).
        # Instead, choose maxexp + 1 if absmax is closer to 2^(maxexp+1) than maxval. This reduces error on
        # the highest magnitude value at the potential cost increased error or underflow of the smallest.
        # Ad hoc MSE test shows that e4m3, due to reserving the most significant value for Nan, benefits the
        # most from this technique.  In hardware or a kernel, this is as simple as comparing bits [30:21]
        # instead of [30:23] when getting max exponent, then add 1 to the max eeeeeeeemm and shift right two.
        #             e2m1    e3m2    e2m3    e4m3    e5m2
        #   max     0.01325 0.00291 0.00080 0.00085 0.00291
        #   best    0.01254 0.00280 0.00079 0.00071 0.00280

        if select:
            midmax = self.midmax * (shared - self.emax).exp2()
            shared[tensor.abs().amax(axis, keepdim=True) > midmax] += 1

        # The way this works is to appropriately shift values so that rounding can work, then shift them back.
        # All values that are representable as normal given the scale are shifted up by the difference
        # between the individual exponent and zero, plus the mantissa width.  Subnormals get the same,
        # but with decreasing mantissa bits.  The maxval for saturation is adjusted on a per block basis.
        scale = (self.mbits - (shared - exp - (self.emax - self.emin)).clamp_min(0) - exp).exp2()
        # about that last line of code:
        # The "offset" is the number of mbits lost to subnormal/underflow. This is based on the difference between
        # the shared exponent and the individual exponent, adjusted to the dynamic range of normals for this type.
        # It can't be negative, because we subtract it from mbits, and don't want to exceed the available mbits.
        #   offset = (shared - exp - (self.emax - self.emin)).clamp_min(0)
        # The shift left will be mbits - offset - exp, which for negative exponents gets them into the right range.
        maxval = self.maxval * (shared - self.emax).exp2()  # scale maxval per tile
        return ((tensor * scale).round() / scale).clamp(-maxval, maxval)


MAP = {"e4m3": (4, 3), "e5m2": (5, 2), "e2m3": (2, 3), "e3m2": (3, 2), "e2m1": (2, 1)}


@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=1e10, min_val=-1e10))
@pytest_cases.parametrize('bit_widths', list(MAP.keys()))
def test_act_mx(inp, bit_widths):
    torch.set_printoptions(precision=12, sci_mode=False)
    exp, mant = MAP[bit_widths]

    act_quant = QuantIdentity(
        MXFloat8e4m3Act,
        exponent_bit_width=exp,
        mantissa_bit_width=mant,
        bit_width=mant + exp + 1,
        group_dim=1,
        return_quant_tensor=True)
    act_quant.eval()
    x = inp

    quantizer = MXFP(bit_widths)

    qx = act_quant(x)

    y = quantizer.quantize(x)
    assert torch.allclose(qx.value, y, atol=1e-8)


@given(inp=float_tensor_nz_st(shape=(1, 32), max_val=1e10, min_val=-1e10))
@pytest_cases.parametrize('bit_widths', list(MAP.keys()))
@pytest_cases.parametrize('weight_quant_type', ['stats', 'parameter_from_stats'])
def test_weight_mx(inp, bit_widths, weight_quant_type):
    torch.set_printoptions(precision=12, sci_mode=False)
    exp, mant = MAP[bit_widths]
    weight_quant = QuantLinear(
        32,
        1,
        bias=False,
        weight_quant=MXFloat8e4m3Weight,
        weight_scaling_impl_type=weight_quant_type,
        weight_exponent_bit_width=exp,
        weight_mantissa_bit_width=mant,
        weight_bit_width=mant + exp + 1)

    x = inp
    weight_quant.weight.data = x
    weight_quant.weight_quant.init_tensor_quant()
    quantizer = MXFP(bit_widths)

    qx_weight = weight_quant.quant_weight()
    qx_weight_two = weight_quant.quant_weight()

    y = quantizer.quantize(x)
    assert torch.allclose(qx_weight.value, y, atol=1e-8)
    assert torch.allclose(qx_weight_two.value, y, atol=1e-8)
