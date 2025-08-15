# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear


class LoRACompatibleQuantConv2d(QuantConv2d):
    """
    A QuantConv2d layer that can be used with as a replacement for LoRACompatibleConv.
    It doesn't actually support LoRA, it only matches the same forward pass.
    """

    def forward(self, hidden_states, scale: float = 1.0):
        return super().forward(hidden_states)


class LoRACompatibleQuantLinear(QuantLinear):
    """
    A QuantLinear layer that can be used with as a replacement for LoRACompatibleLinear.
    It doesn't actually support LoRA, it only matches the same forward pass.
    """

    def forward(self, hidden_states, scale: float = 1.0):
        return super().forward(hidden_states)
