# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Implementation of AutoRound
"""

from typing import Optional

import torch

import brevitas
from brevitas import config
from brevitas.core.utils import SliceTensor
from brevitas.function.ops_ste import round_ste


class AutoRoundSte(brevitas.jit.ScriptModule):
    """
    This Module implements AutoRound representation, where each weight has a learnable parameter
    that decides if "ceil" or "floor" rounding type has to be used.
    """

    def __init__(
            self,
            learned_round_init: torch.Tensor,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None) -> None:
        super(AutoRoundSte, self).__init__()
        learned_round_init = learned_round_init.to(device=device, dtype=dtype)
        self.tensor_slicer = SliceTensor()
        self.value = torch.nn.Parameter(learned_round_init)

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # p should be between [-0.5, 0.5], so this learnable parameter decides whether to "ceil" or "floor"
        p = self.value
        p = self.tensor_slicer(p)
        return round_ste(x + p.to(x.dtype))

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(AutoRoundSte, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + 'value'
        if config.IGNORE_MISSING_KEYS and value_key in missing_keys:
            missing_keys.remove(value_key)
