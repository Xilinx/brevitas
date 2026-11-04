# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import torch
from torch import nn
from torch import Tensor

import brevitas
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int


class IntScaling(brevitas.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(
            self,
            narrow_range: bool,
            signed: Optional[bool] = None,
            dtype: Optional[torch.dtype] = None):
        super(IntScaling, self).__init__()
        self.signed = signed
        self.narrow_range = narrow_range
        self.dtype = dtype

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor, signed: Optional[Union[bool, Tensor]] = None) -> Tensor:
        is_signed = signed if signed is not None else self.signed
        if is_signed is None:
            raise ValueError(f"signed is not defined, signed={is_signed}")
            is_signed = True  # Workaround for JIT type inference
        # Workaround: required for compatibility with the JIT for PT=2.2.2
        is_signed = bool(is_signed.item()) if isinstance(is_signed, Tensor) else is_signed
        if is_signed:
            return -min_int(is_signed, self.narrow_range, bit_width).to(dtype=self.dtype)
        else:
            return max_int(is_signed, self.narrow_range, bit_width).to(dtype=self.dtype)


class PowerOfTwoIntScaling(brevitas.jit.ScriptModule):
    """
    Integer threshold for power-of-two scaling: 2^(bit_width - 1) when signed, 2^bit_width when
    unsigned (i.e., max_int + 1, regardless of narrow_range), multiplied by
    2^int_scaling_exponent_offset.

    The default offset of 0 pairs with scales rounded up to a power of two (fixed-point), which
    puts amax / scale in (2^(bit_width - 2), 2^(bit_width - 1)] for signed values. Scales rounded
    down to a power of two (e.g., the shared exponent of OCP MX) need an offset of -1 to leave
    headroom for the top binade, so that amax / scale lies in [2^(bit_width - 2), 2^(bit_width - 1)).

    Args:
        signed (bool, optional): Whether the integer range is signed. Can be overridden at runtime.
        int_scaling_exponent_offset (int): Power-of-two exponent added to the threshold. Default: 0
    """
    __constants__ = ['signed', 'int_scaling_exponent_offset']

    def __init__(self, signed: Optional[bool] = None, int_scaling_exponent_offset: int = 0):
        super(PowerOfTwoIntScaling, self).__init__()
        self.signed = signed
        self.int_scaling_exponent_offset = int_scaling_exponent_offset

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor, signed: Optional[Union[bool, Tensor]] = None) -> Tensor:
        is_signed = signed if signed is not None else self.signed
        if is_signed is None:
            raise ValueError(f"signed is not defined, signed={is_signed}")
            is_signed = True  # Workaround for JIT type inference
        # Workaround: required for compatibility with the JIT for PT=2.2.2
        is_signed = bool(is_signed.item()) if isinstance(is_signed, Tensor) else is_signed
        return (max_int(is_signed, False, bit_width) + 1) * 2.0 ** self.int_scaling_exponent_offset
