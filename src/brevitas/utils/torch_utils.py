# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import dataclass
from functools import wraps
from typing import List, Optional, Tuple

import torch
from torch.nn import Sequential

import brevitas
import brevitas.compiler as brevitas_compiler
from brevitas.function.ops_ste import floor_ste


class StopFwdException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


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
@brevitas_compiler.disable
def padding_to_multiple(x: torch.Tensor, dim_to_expand: int, dim_multiple: int) -> torch.Tensor:
    # Given a tensor X, compute the padding along dim_multiple so that new dimension is a multiple of dim_multiple
    padding = [0, 0] * len(x.shape)
    size = x.shape
    if size[dim_to_expand] % dim_multiple != 0:
        padding[2 * dim_to_expand] = dim_multiple - size[dim_to_expand] % dim_multiple
    padding = list(reversed(padding))
    x = torch.nn.functional.pad(x, padding, mode='constant', value=0.)
    return x


def pad_to_dim(tensor: torch.Tensor, dim_to_expand: int, new_dim: int) -> torch.Tensor:
    # Given a tensor X, compute the padding along dim_to_expand so that we match the desired new_dim
    current_dim = tensor.shape[dim_to_expand]
    pad_size = int(new_dim - current_dim)
    pad_dim = len(tensor.data.shape) * 2
    pad_tensor = [0] * pad_dim
    pad_tensor[dim_to_expand * 2] = pad_size
    pad_tensor = list(reversed(pad_tensor))
    new_tensor = torch.nn.functional.pad(tensor, pad_tensor)
    return new_tensor


def is_broadcastable(tensor, other):
    for a, b in zip(tensor[::-1], other[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def torch_dtype(dtype):

    def decorator(fn):

        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            cur_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                fn(*args, **kwargs)
            finally:
                torch.set_default_dtype(cur_dtype)

        return wrapped_fn

    return decorator


# TODO: Remove after deprecating PyTorch 1.11
def is_parametrized(module: torch.nn.Module, tensor_name: Optional[str] = None) -> bool:
    r"""Determine if a module has a parametrization.

    Args:
        module (nn.Module): module to query
        tensor_name (str, optional): name of the parameter in the module
            Default: ``None``
    Returns:
        ``True`` if :attr:`module` has a parametrization for the parameter named :attr:`tensor_name`,
        or if it has any parametrization when :attr:`tensor_name` is ``None``;
        otherwise ``False``
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, torch.nn.ModuleDict):
        return False
    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(parametrizations) > 0
    else:
        return tensor_name in parametrizations


# TODO: Remove after deprecating PyTorch 1.11
def type_before_parametrizations(module: torch.nn.Module) -> type:
    r"""Return the module type before parametrizations were applied and if not, then it returns the module type.

    Args:
        module (nn.Module): module to get type of
    """
    if is_parametrized(module):
        return module.__class__.__bases__[0]
    else:
        return type(module)
