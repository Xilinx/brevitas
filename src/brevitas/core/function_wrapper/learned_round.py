# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Different implementations for LearnedRound.
"""

from typing import Optional

import torch

import brevitas
from brevitas import config
from brevitas.core.function_wrapper.ops_ste import TensorClampSte
from brevitas.core.utils import SliceTensor
from brevitas.function.ops_ste import floor_ste
from brevitas.function.ops_ste import round_ste


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
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(p)
        p = p * (self.learned_round_zeta - self.learned_round_gamma) + self.learned_round_gamma
        p = torch.clamp(p, 0.0, 1.0)
        if not self.training:
            return p > 0.5
        return p

    def round_forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return floor_ste(x) + p


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
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return p > 0
        p = torch.sigmoid(p / self.learned_round_temperature)
        return p

    @brevitas.jit.script_method
    def round_forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return floor_ste(x) + p


class LearnedRoundIdentity(brevitas.jit.ScriptModule):
    """
    Implementation for LearnedRound learned parameter
    Adapted from https://arxiv.org/abs/2309.05516
    """

    def __init__(self) -> None:
        super(LearnedRoundIdentity, self).__init__()
        self.tensor_clamp = TensorClampSte()
        self.upper_lower_bound = brevitas.jit.Attribute(0.5, float)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.tensor_clamp(
            p,
            min_val=torch.tensor(-self.upper_lower_bound).type_as(p),
            max_val=torch.tensor(self.upper_lower_bound).type_as(p))

    def round_forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return round_ste(x + p)


class LearnedRoundSte(brevitas.jit.ScriptModule):
    """
    This Module implements LearnedRound representation, where each weight has a learnable parameter
    that decides if "ceil" or "floor" rounding type has to be used.
    """

    def __init__(
            self,
            learned_round_impl: torch.nn.Module,
            learned_round_init: torch.Tensor,
            input_view_impl: torch.nn.Module,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None) -> None:
        super(LearnedRoundSte, self).__init__()
        self.learned_round_impl = learned_round_impl
        learned_round_init = learned_round_init.to(device=device, dtype=dtype)
        self.tensor_slicer = SliceTensor()
        self.value = torch.nn.Parameter(learned_round_init)
        self.input_view_impl = input_view_impl

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.learned_round_impl(self.value)
        p = self.input_view_impl(p)
        p = self.tensor_slicer(p)
        return self.learned_round_impl.round_forward(x, p)

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(LearnedRoundSte, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)
