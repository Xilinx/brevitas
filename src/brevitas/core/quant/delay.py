# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from torch import Tensor

import brevitas


class _NoDelay(brevitas.jit.ScriptModule):

    @brevitas.jit.script_method
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return y


class _DelayQuant(brevitas.jit.ScriptModule):

    def __init__(self, quant_delay_steps):
        super(_DelayQuant, self).__init__()
        self.quant_delay_steps: int = brevitas.jit.Attribute(quant_delay_steps, int)

    @brevitas.jit.script_method_110_disabled
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.quant_delay_steps > 0:
            self.quant_delay_steps = self.quant_delay_steps - 1
            return x
        else:
            return y

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(_DelayQuant, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # Pytorch stores training flag as a buffer with JIT enabled
        training_key = prefix + 'training'
        if training_key in missing_keys:
            missing_keys.remove(training_key)


class DelayWrapper(brevitas.jit.ScriptModule):

    def __init__(self, quant_delay_steps: Optional[int]):
        super(DelayWrapper, self).__init__()
        if quant_delay_steps is None or quant_delay_steps <= 0:
            self.delay_impl = _NoDelay()
        else:
            self.delay_impl = _DelayQuant(quant_delay_steps)

    @brevitas.jit.script_method
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.delay_impl(x, y)
