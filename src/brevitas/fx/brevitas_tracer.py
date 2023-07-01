# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import ExitStack
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.nn import Sequential

from brevitas.utils.python_utils import patch

from . import GraphModule
from . import map_aggregate
from . import Proxy
from . import Tracer
from .value_tracer import _UNSET
from .value_tracer import UnsetValueException
from .value_tracer import ValueTracer

FNS_TO_PATCH = [
    torch.arange,
    torch.tensor,
    torch.zeros,
    torch.ones,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.full,
    torch.finfo]


def _gen_torch_fn_patches(orig_fn):
    tracers: Dict[Any, None] = {}

    def find_tracer(a):
        if isinstance(a, Proxy):
            tracers[a.tracer] = None

    def new_fn(*args, **kwargs):
        map_aggregate(args, find_tracer)
        map_aggregate(kwargs, find_tracer)

        # In case no tracer has been found, return the output of orig_fn as constant value to be
        # tracked by following ops
        if len(tracers) == 0:
            return orig_fn(*args, **kwargs)

        assert len(tracers) == 1, "Multiple different tracers found."

        tracer = next(iter(tracers.keys()))
        try:
            value = orig_fn(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
        except UnsetValueException:
            value = _UNSET
        return tracer.create_proxy(
            'call_function',
            orig_fn,
            args,
            kwargs,
            name=tracer.graph._target_to_str(orig_fn.__name__),
            value=value)

    patch_fn = patch(torch, orig_fn.__name__, new_fn)
    return patch_fn


def _gen_patches():

    original_torch_cat = torch.cat
    original_torch_stack = torch.stack

    def cat(tensors, dim, out=None):
        if not isinstance(tensors, (tuple, list)):
            tensors = tuple(tensors)
        if out is not None:
            return original_torch_cat(tensors, dim, out)
        else:
            return original_torch_cat(tensors, dim)

    def stack(tensors, dim, out=None):
        if not isinstance(tensors, (tuple, list)):
            tensors = tuple(tensors)
        if out is not None:
            return original_torch_stack(tensors, dim, out)
        else:
            return original_torch_stack(tensors, dim)

    cat_patch = patch(torch, 'cat', cat)
    stack_patch = patch(torch, 'stack', stack)

    tensor_creation_patches = []
    for fn in FNS_TO_PATCH:
        tensor_creation_patches.append(_gen_torch_fn_patches(fn))

    return [cat_patch, stack_patch] + tensor_creation_patches


def _is_brevitas_leaf_module(m, fully_qualified_name):
    is_torch_nn = m.__module__.startswith('torch.nn')
    is_brevitas_nn = m.__module__.startswith('brevitas.nn')
    is_brevitas_core = m.__module__.startswith('brevitas.core')
    is_brevitas_proxy = m.__module__.startswith('brevitas.proxy')
    is_seq = isinstance(m, Sequential)
    is_brevitas = is_brevitas_nn or is_brevitas_proxy or is_brevitas_core
    return is_torch_nn and not is_seq or is_brevitas


def _symbolic_trace(
        tracer: Tracer, root: Union[Module, Callable],
        concrete_args: Optional[Dict[str, Any]]) -> GraphModule:
    patches = _gen_patches()
    if patches:
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


def _value_trace(
        tracer: Tracer,
        root: Union[Module, Callable],
        concrete_args: Optional[Dict[str, Any]],
        value_args: Optional[Dict[str, Any]]) -> GraphModule:
    patches = _gen_patches()
    if patches:
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            graph = tracer.trace(root, concrete_args, value_args)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


class BrevitasValueTracer(ValueTracer):

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return _is_brevitas_leaf_module(m, module_qualified_name)


class BrevitasSymbolicTracer(Tracer):

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return _is_brevitas_leaf_module(m, module_qualified_name)


def symbolic_trace(root, concrete_args=None):
    return _symbolic_trace(Tracer(), root, concrete_args)


def value_trace(root, concrete_args=None, value_args=None):
    return _value_trace(ValueTracer(), root, concrete_args, value_args)


def brevitas_symbolic_trace(root, concrete_args=None):
    return _symbolic_trace(BrevitasSymbolicTracer(), root, concrete_args)


def brevitas_value_trace(root, concrete_args=None, value_args=None):
    return _value_trace(BrevitasValueTracer(), root, concrete_args, value_args)
