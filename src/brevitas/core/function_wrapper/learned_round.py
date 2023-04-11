# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Different implementations for LearnedRound.
"""

import torch

import brevitas
from brevitas import config
from brevitas.function.ops_ste import floor_ste


class LearnedRoundHardSigmoid(brevitas.jit.ScriptModule):
    """
    HardSigmoid implementation for LearnedRound learned parameter
    Adapted from https://arxiv.org/abs/2004.10568
    """
    __constants__ = ['learned_round_zeta', 'learned_round_gamma']

    def __init__(self, learned_round_zeta: float = 1.1, learned_round_gamma: float = -0.1) -> None:
        super(LearnedRoundHardSigmoid, self).__init__()
        self.learned_round_zeta = learned_round_zeta
        self.learned_round_gamma = learned_round_gamma

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x)
        p = p * (self.learned_round_zeta - self.learned_round_gamma) + self.learned_round_gamma
        p = torch.clamp(p, 0.0, 1.0)
        return p


class LearnedRoundSigmoid(brevitas.jit.ScriptModule):
    """
    Sigmoid implementation for LearnedRound learned parameter. Supports for temperature factor
    """
    __constants__ = ['learned_round_temperature']

    def __init__(self, learned_round_temperature: float = 1.) -> None:
        super(LearnedRoundSigmoid, self).__init__()
        assert learned_round_temperature != 0, 'Temperature should be different than 0'
        self.learned_round_temperature = learned_round_temperature

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(x / self.learned_round_temperature)
        return p


class LearnedRoundSte(brevitas.jit.ScriptModule):
    """
    This Module implements LearnedRound representation, where each weight has a learnable parameter
    that decides if "ceil" or "floor" rounding type has to be used.
    """

    def __init__(
            self, learned_round_impl: torch.nn.Module, learned_round_init: torch.Tensor) -> None:
        super(LearnedRoundSte, self).__init__()
        self.learned_round_impl = learned_round_impl
        self.value = torch.nn.Parameter(learned_round_init)

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p_forward()
        return floor_ste(x) + p.to(x.dtype)

    def p_forward(self):
        # In eval mode, performs true quantization, otherwise "soft" quantization
        if not self.training:
            p = (self.value > 0)
        else:
            p = self.learned_round_impl(self.value)
        return p

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(LearnedRoundSte, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)
