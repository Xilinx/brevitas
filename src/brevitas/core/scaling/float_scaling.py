# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple

import torch
from torch import Tensor

import brevitas
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_float


class FloatScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            max_available_float: Optional[float] = None,
            inf_values: Optional[Tuple[str]] = None,
            nan_values: Optional[Tuple[str]] = None,
            saturating: bool = True,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None):
        super(FloatScaling, self).__init__()
        self.inf_values = inf_values
        self.nan_values = nan_values
        self.saturating = saturating
        self.dtype = dtype

        if max_available_float:
            max_available_float = torch.tensor(max_available_float, device=device, dtype=dtype)
            self.max_available_float = StatelessBuffer(max_available_float)
        else:
            self.max_available_float = None

    @brevitas.jit.script_method
    def forward(
            self, exponent_bit_width: Tensor, pre_compute_max_mantissa: Tensor,
            exponent_bias: Tensor) -> Tensor:
        max_value = max_float(exponent_bit_width, pre_compute_max_mantissa,
                              exponent_bias).to(self.dtype)
        max_value = max_value if self.max_available_float is None else torch.min(
            max_value, self.max_available_float())
        return max_value
