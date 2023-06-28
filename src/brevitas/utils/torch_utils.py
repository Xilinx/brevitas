# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch
from torch.nn import Sequential


class TupleSequential(Sequential):

    def output(self, mod, input):
        if isinstance(input, tuple):
            return mod(*input)
        else:
            return mod(input)

    def forward(self, *input):
        modules = list(self._modules.values())
        out = self.output(modules[0], input)
        for mod in modules[1:]:
            out = self.output(mod, out)
        return out


class KwargsForwardHook(torch.nn.Module):

    def __init__(self, module, hook_fn):
        super().__init__()
        self.module = module
        self.hook_fn = hook_fn

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        args = args + (out,)
        self.hook_fn(self.module, *args, **kwargs)
        return out


def torch_partial_deepcopy(model):
    """
    Performs a deepcopy of a torch.nn.Module, except for all the parameters that are instead passed by reference
    """
    memo = {}
    for p in model.parameters():
        memo[id(p)] = copy.copy(p)  # Shallow copy of parameters
    model_copy = copy.deepcopy(model, memo)
    return model_copy
