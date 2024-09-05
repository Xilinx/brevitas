# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import List, Optional, Tuple

import torch
from torch.nn import Sequential

import brevitas
from brevitas.function.ops_ste import floor_ste


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


def kthvalue(
    x: torch.Tensor,
    k: int,
    dim: Optional[int] = None,
    keepdim: bool = False,
    out: Optional[Tuple[torch.Tensor, torch.LongTensor]] = None
) -> Tuple[torch.Tensor, torch.LongTensor]:
    # As of torch 2.1, there is no kthvalue implementation:
    # - In CPU for float16
    # - In GPU for bfloat16
    # In these cases we cast to float32 and then go back to the original dtype
    dtype = x.dtype
    device = str(x.device)

    # We do not support out as buffer for the output, since we cannot control its dtype
    if out is not None:
        raise RuntimeError("out argument for kthvalue not supported")

    if (dtype == torch.float16 and 'cpu' in device) or \
        (dtype == torch.bfloat16 and 'cuda' in device):
        x = x.type(torch.float32)

    # PyTorch specify None as default for `dim` but it breaks if we specifically pass None
    if dim is not None:
        x, indices = torch.kthvalue(x, k, dim=dim, keepdim=keepdim)
    else:
        x, indices = torch.kthvalue(x, k, keepdim=keepdim)

    if x.dtype != dtype:
        x = x.type(dtype)
    return (x, indices)


def compute_channel_view_shape(tensor: torch.Tensor, channel_dim: int):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)


@brevitas.jit.script
def float_internal_scale(
        x: torch.Tensor,
        mantissa_bit_width: torch.Tensor,
        fp_internal_scale_min: torch.Tensor,
        eps: float) -> torch.Tensor:

    internal_scale = floor_ste(torch.log2(torch.abs(x) + eps)) - mantissa_bit_width
    internal_scale = torch.clamp_min(internal_scale, fp_internal_scale_min)
    internal_scale = torch.exp2(internal_scale)
    return internal_scale


@brevitas.jit.ignore
def padding(x: torch.Tensor, group_size: int, group_dim: int) -> List[int]:
    # Given a tensor X, compute the padding aloing group_dim so that groupwise shaping is possible
    padding = [0, 0] * len(x.shape)
    size = x.shape
    if size[group_dim] % group_size != 0:
        padding[2 * group_dim] = group_size - size[group_dim] % group_size
    padding = list(reversed(padding))
    return padding
