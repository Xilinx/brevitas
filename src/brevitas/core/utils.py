# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch

import brevitas

VALUE_ATTR_NAME = 'value'


@torch.jit.ignore
def inplace_tensor_add(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    tensor.add_(value)
    return tensor


@torch.jit.ignore
def inplace_tensor_mul(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    tensor.mul_(value)
    return tensor


@torch.jit.ignore
def inplace_momentum_update(
        tensor: torch.Tensor,
        update: torch.Tensor,
        momentum: Optional[float],
        counter: int,
        new_counter: int) -> torch.Tensor:
    if momentum is None:
        tensor.mul_(counter / new_counter)
        tensor.add_(update / new_counter)
    else:
        tensor.mul_(1 - momentum)
        tensor.add_(momentum * update)
    return tensor


class StatelessBuffer(brevitas.jit.ScriptModule):

    def __init__(self, value: torch.Tensor):
        super(StatelessBuffer, self).__init__()
        self.register_buffer(VALUE_ATTR_NAME, value)

    @brevitas.jit.script_method
    def forward(self):
        return self.value.detach()

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(StatelessBuffer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        value_key = prefix + VALUE_ATTR_NAME
        if value_key in missing_keys:
            missing_keys.remove(value_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(StatelessBuffer, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        del output_dict[prefix + VALUE_ATTR_NAME]
        return output_dict


class SingleArgStatelessBuffer(brevitas.jit.ScriptModule):

    def __init__(self, value: torch.Tensor):
        super(SingleArgStatelessBuffer, self).__init__()
        self.const = StatelessBuffer(torch.tensor(value))

    @brevitas.jit.script_method
    def forward(self, placeholder):
        return self.const()


class ParameterWrapper(brevitas.jit.ScriptModule):

    def __init__(self, value: torch.Tensor):
        super(ParameterWrapper, self).__init__()
        self.register_parameter(VALUE_ATTR_NAME, value)

    @brevitas.jit.script_method
    def forward(self):
        return self.value
