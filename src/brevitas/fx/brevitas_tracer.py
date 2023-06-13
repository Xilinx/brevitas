# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import ExitStack
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.nn import Sequential

from brevitas.utils.python_utils import patch

from . import GraphModule
from . import Tracer
from .value_tracer import ValueTracer


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

    return [cat_patch, stack_patch]


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
