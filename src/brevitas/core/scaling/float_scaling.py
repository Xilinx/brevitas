# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor

import brevitas
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_float


class FloatScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            exponent_bit_width: int,
            mantissa_bit_width: int,
            exponent_bias: int,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None):
        super(FloatScaling, self).__init__()
        exponent_bit_width = torch.tensor(exponent_bit_width, device=device, dtype=dtype)
        mantissa_bit_width = torch.tensor(mantissa_bit_width, device=device, dtype=dtype)
        exponent_bias = torch.tensor(exponent_bias, device=device, dtype=dtype)
        self.max_float_val = StatelessBuffer(
            max_float(exponent_bit_width, mantissa_bit_width, exponent_bias))

    @brevitas.jit.script_method
    def forward(self, input: torch.Tensor) -> Tensor:
        return self.max_float_val()
