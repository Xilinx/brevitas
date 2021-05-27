from typing import Callable, Optional, Dict, Any, Union

from contextlib import ExitStack
from torch.nn import Sequential, Module

from . import GraphModule, Tracer
from .value_tracer import ValueTracer
from .backport.torch_function import gen_patches


def _is_brevitas_leaf_module(m, fully_qualified_name):
    is_torch_nn = m.__module__.startswith('torch.nn')
    is_brevitas_nn = m.__module__.startswith('brevitas.nn')
    is_brevitas_core = m.__module__.startswith('brevitas.core')
    is_brevitas_proxy = m.__module__.startswith('brevitas.proxy')
    is_seq = isinstance(m, Sequential)
    is_brevitas = is_brevitas_nn or is_brevitas_proxy or is_brevitas_core
    return is_torch_nn and not is_seq or is_brevitas


def _trace_with_backport(
        tracer: Tracer,
        root : Union[Module, Callable],
        concrete_args: Optional[Dict[str, Any]]) -> GraphModule:
    patches = gen_patches()
    if patches:
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            graph = tracer.trace(root, concrete_args)
    else:
        graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


class BrevitasValueTracer(ValueTracer):

    def is_leaf_module(self, m: Module, module_qualified_name : str) -> bool:
        return _is_brevitas_leaf_module(m, module_qualified_name)


class BrevitasSymbolicTracer(Tracer):

    def is_leaf_module(self, m: Module, module_qualified_name : str) -> bool:
        return _is_brevitas_leaf_module(m, module_qualified_name)


def symbolic_trace(root, concrete_args = None):
    return _trace_with_backport(Tracer(), root, concrete_args)


def value_trace(root, concrete_args = None):
    return _trace_with_backport(ValueTracer(), root, concrete_args)


def brevitas_symbolic_trace(root, concrete_args = None):
    return _trace_with_backport(BrevitasSymbolicTracer(), root, concrete_args)


def brevitas_value_trace(root, concrete_args = None):
    return _trace_with_backport(BrevitasValueTracer(), root, concrete_args)





