# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

from torch import Tensor

import brevitas
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int


class IntScaling(brevitas.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, narrow_range: bool, signed: Optional[bool] = None):
        super(IntScaling, self).__init__()
        self.signed = signed
        self.narrow_range = narrow_range

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor, signed: Optional[Union[bool, Tensor]] = None) -> Tensor:
        is_signed = signed if signed is not None else self.signed
        assert is_signed is not None, f"signed is not defined, signed={is_signed}"
        if is_signed:
            return -min_int(is_signed, self.narrow_range, bit_width)
        else:
            return max_int(is_signed, self.narrow_range, bit_width)


class PowerOfTwoIntScaling(brevitas.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed: Optional[bool] = None):
        super(PowerOfTwoIntScaling, self).__init__()
        self.signed = signed

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor, signed: Optional[Union[bool, Tensor]] = None) -> Tensor:
        is_signed = signed if signed is not None else self.signed
        assert is_signed is not None, f"signed is not defined, signed={is_signed}"
        return max_int(is_signed, False, bit_width) + 1
