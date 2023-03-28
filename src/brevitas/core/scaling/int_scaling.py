# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor

import brevitas
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int


class IntScaling(brevitas.jit.ScriptModule):
    __constants__ = ['signed', 'narrow_range']

    def __init__(self, signed: bool, narrow_range: bool):
        super(IntScaling, self).__init__()
        self.signed = signed
        self.narrow_range = narrow_range

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor) -> Tensor:
        if self.signed:
            return -min_int(self.signed, self.narrow_range, bit_width)
        else:
            return max_int(self.signed, self.narrow_range, bit_width)


class PowerOfTwoIntScaling(brevitas.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(self, signed: bool):
        super(PowerOfTwoIntScaling, self).__init__()
        self.signed = signed

    @brevitas.jit.script_method
    def forward(self, bit_width: Tensor) -> Tensor:
        return max_int(self.signed, False, bit_width) + 1
