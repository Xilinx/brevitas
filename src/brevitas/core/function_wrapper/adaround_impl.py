# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Different implementations for adaround parameter transformation
"""

import torch

import brevitas


class AdaRoundHardSigmoid(brevitas.jit.ScriptModule):
    """
    HardSigmoid implementation for AdaRound learned parameter
    """
    __constants__ = ['zeta', 'gamma']

    def __init__(self, zeta: float = 1.1, gamma: float = -0.1) -> None:
        super(AdaRoundHardSigmoid, self).__init__()
        self.zeta = zeta
        self.gamma = gamma

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x)
        p = torch.clamp(p * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
        return p


class AdaRoundSigmoid(brevitas.jit.ScriptModule):
    """
    Sigmoid implementation for AdaRound learned parameter. Supports for temperature factor
    """
    __constants__ = ['adaround_temperature']

    def __init__(self, adaround_temperature: float = 1.) -> None:
        super(AdaRoundHardSigmoid, self).__init__()
        assert adaround_temperature != 0, 'Temperature should be different than 0'
        self.adaround_temperature = adaround_temperature

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x / self.adaround_temperature)
        return p
