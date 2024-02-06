"""
Copyright (c) 2023-     Advanced Micro Devices, Inc. (Alessandro Pappalardo)
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import builtins
import functools
import inspect
import math
import operator
import os
import traceback
from types import FunctionType
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import warnings

from packaging import version
from torch._C import ScriptObject
import torch.utils._pytree as pytree

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensorBase

from . import *
from . import _assert_is_none
from . import _autowrap_check
from . import _find_proxy
from . import _is_fx_tracing_flag
from . import _orig_module_call
from . import _orig_module_getattr
from . import _patch_function
from . import _Patcher
from . import _proxyable_classes
from . import _PyTreeCodeGen
from . import _PyTreeInfo
from . import _wrapped_fns_to_patch
from . import _wrapped_methods_to_patch
from . import compatibility
from . import fx
from . import fx_traceback
from . import ParameterProxy
from . import PH
from . import Scope
from . import ScopeContextManager

_UNSET = object()
extended_base_types = base_types + (QuantTensorBase,)

FRAME_FILES = [
    'fx/brevitas_tracer.py',
    'fx/value_tracer.py',
    'fx/proxy.py',
    'fx/node.py',
    'fx/_symbolic_trace.py',
    'fx/experimental/proxy_tensor.py',
    'fx/graph_module.py',
    'torch/_ops.py',
    'torch/_tensor.py',
    'torch/nn/parameter.py',
    'torch/nn/modules/module.py',
    'torch/utils/_python_dispatch.py',
    'torch/_prims_common/wrappers.py',
    'torch/_refs/__init__.py',
    'torch/_refs/nn/functional/__init__.py']


class UnsetValueException(Exception):
    pass


# https://github.com/python/cpython/issues/56579
_safe_super = super


@compatibility(is_backward_compatible=True)
class ValueProxy(Proxy):

    def __init__(self, node: Node, value, tracer=None):
        _safe_super(ValueProxy, self).__init__(node, tracer)
        if isinstance(value, Proxy):
            raise RuntimeError("Value of a proxy can't be a proxy.")
        self.value = value

    def __index__(self):
        return self.value.__index__()

    @property
    def value(self):
        if self._value is _UNSET:
            raise UnsetValueException
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def __class__(self):
        previous_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(previous_frame).filename
        for f in FRAME_FILES:
            if os.path.normpath(f) in os.path.normpath(filename):
                return type(self)
        return self.value.__class__

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, type(self))

    def __subclasscheck__(self, subclass: type) -> bool:
        return issubclass(subclass, type(self))

    def __getattr__(self, k) -> 'ValueAttribute':
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        try:
            value = getattr(self.value, k)
        except UnsetValueException:
            value = _UNSET
        if value is None:
            return None
        return ValueAttribute(self, k, value)

    def __call__(self, *args, **kwargs) -> 'Proxy':
        try:
            value = self.value(*self.unpack_arg(args), **self.tracer.unpack_arg(kwargs))
        except UnsetValueException:
            value = _UNSET
        return self.tracer.create_proxy(
            'call_method', '__call__', (self,) + args, kwargs, value=value)

    def __len__(self):
        return self.value.__len__()

    def __next__(self):
        return self.tracer.create_proxy(
            'call_method', '__next__', (self,), {}, value=self.value.__next__())

    def __setitem__(self, key, item_value):
        value = self.value.__setitem__(key, item_value)
        return self.tracer.create_proxy(
            'call_method', '__setitem__', (self, key, item_value), {}, value=value)

    def __setslice__(self, i, j, sequence):
        value = self.value.__setitem__(i, j, sequence)
        return self.tracer.create_proxy(
            'call_method', '__setitem__', (self, i, j, sequence), {}, value=value)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None

        fx.node.map_aggregate(args, find_tracer)
        fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(
                f'Found multiple different tracers {list(tracers.keys())} while '
                f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        def retrieve_method_proxy(method_args, method_kwargs):
            # Because __torch_function__ is a cls method,
            # we need to find self in the args or kwargs
            proxies: List[Proxy] = []

            def find_proxy(a):
                if isinstance(a, cls):
                    proxies.append(a)

            fx.node.map_aggregate(method_args, find_proxy)
            fx.node.map_aggregate(method_kwargs, find_proxy)
            assert len(proxies) == 1, "Method call expect a single source proxy."
            return proxies[0]

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            proxy = retrieve_method_proxy(args, kwargs)
            try:
                value = proxy.value(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs, value=value)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            proxy = retrieve_method_proxy(args, kwargs)
            try:
                value = orig_method(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return tracer.create_proxy(
                'call_method', orig_method.__name__, args, kwargs, value=value)
        else:
            try:
                value = orig_method(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return tracer.create_proxy(
                'call_function',
                orig_method,
                args,
                kwargs,
                name=tracer.graph._target_to_str(orig_method.__name__),
                value=value)


@compatibility(is_backward_compatible=True)
class ValueAttribute(ValueProxy):

    def __init__(self, root: Proxy, attr: str, value: Any):
        self.root = root
        self.attr = attr
        self.value = value
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def __class__(self):
        previous_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(previous_frame).filename
        for f in FRAME_FILES:
            if f in filename:
                return type(self)
        return self.value.__class__

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            proxy = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {})
            self._node = proxy.node
        return self._node

    def __call__(self, *args, **kwargs):
        try:
            value = self.value(*self.tracer.unpack_arg(args), **self.tracer.unpack_arg(kwargs))
        except UnsetValueException:
            value = _UNSET
        return self.tracer.create_proxy(
            'call_method', self.attr, (self.root,) + args, kwargs, value=value)


for method in magic_methods:

    def _scope(method):

        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            try:
                value = target(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return tracer.create_proxy('call_function', target, args, kwargs, value=value)

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(ValueProxy, as_magic, impl)

    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        try:
            value = target(self.tracer.unpack_arg(rhs), self.value)
        except UnsetValueException:
            value = _UNSET
        return self.tracer.create_proxy('call_function', target, (rhs, self), {}, value=value)

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(ValueProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)


class ValueTracer(Tracer):

    def __init__(self, autowrap_modules: Tuple[ModuleType] = (math,)):
        super(ValueTracer, self).__init__(autowrap_modules)
        self.concrete_mode = False

    def to_bool(self, obj: 'Proxy') -> bool:
        return obj.value.__bool__()

    def iter(self, obj: 'Proxy'):
        return self.create_proxy('call_function', iter, (obj,), {}, value=obj.value.__iter__())

    # TODO deal with keys and values
    def keys(self, obj: 'Proxy') -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return ValueAttribute(obj, 'keys')()

    def proxy(self, node: Node, value: Any) -> 'Proxy':
        # We don't want to proxy None, or it will affect
        # the correctness of the tracing process for is None checks
        if value is None:
            return None
        return ValueProxy(node, value, self)

    def create_arg(self, a: Any):
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        # Tracer implementation below

        # The base tracer is used to construct Graphs when there is no associated
        # module hierarchy, so it can never create parameter references.
        # The default tracer adds the ability to refer to parameters when
        # tracing modules.
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        # Modified to account for QuantTensor
        if isinstance(a, tuple) and hasattr(a, "_fields") and not isinstance(extended_base_types):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # special attribute and set the qualname to refer to that
            if not qualname:
                i = 0
                while True:
                    qualname = f"_tensor_constant{i}"
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            i = 0
            while True:
                qualname = f"_{a.__class__.__name__}_constant_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        # TracerBase implementation below
        if not isinstance(a, Proxy) and hasattr(a, '__fx_create_arg__'):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple) and hasattr(a,
                                              '_fields') and not isinstance(a, extended_base_types):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.create_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore[arg-type]
        elif isinstance(a, (tuple, list)) and not isinstance(a, extended_base_types):
            return type(a)(self.create_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # Check for invalid dict keys. We do not want a Proxy to appear
                # anywhere within the key. Since keys can be collection types,
                # we iterate through the key with map_aggregate
                k = self.create_arg(k)

                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError(
                            "Keys for dictionaries used as an argument cannot contain a "
                            f"Node. Got key: {k}")

                map_aggregate(k, no_node)

                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))

        elif isinstance(a, range):
            return range(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))

        if isinstance(a, Proxy):
            # base case: we unwrap the Proxy object
            return a.node
        # modified to check against extended_base_types
        elif isinstance(a, extended_base_types) or a is None or a is ...:
            return a
        raise NotImplementedError(f"argument of type: {type(a)}")

    def unpack_arg(self, a: Any):
        """
        This method is based on create_arg but it instead unpacks the value
        out of a ValueProxy, rather than retrieve a node.
        """
        if isinstance(a, tuple) and hasattr(a,
                                            '_fields') and not isinstance(a, extended_base_types):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.unpack_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore
        elif isinstance(a, (tuple, list)) and not isinstance(a, extended_base_types):
            return type(a)(self.unpack_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # Check for invalid dict keys. We do not want a Proxy to appear
                # anywhere within the key. Since keys can be collection types,
                # we iterate through the key with map_aggregate
                k = self.unpack_arg(k)

                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError(
                            "Keys for dictionaries used as an argument cannot contain a Node. "
                            "Got key: {k}")

                map_aggregate(k, no_node)

                r[k] = self.unpack_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(self.unpack_arg(a.start), self.unpack_arg(a.stop), self.unpack_arg(a.step))

        if isinstance(a, ValueProxy):
            # base case: we unpack the value
            return a.value
        elif isinstance(a, extended_base_types) or a is None or a is ...:
            return a

        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def create_proxy(
            self,
            kind: str,
            target: Target,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            name: Optional[str] = None,
            type_expr: Optional[Any] = None,
            proxy_factory_fn: Callable[[Node], 'Proxy'] = None,
            value: Any = _UNSET):
        '''
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        '''

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node, value)
        else:
            proxy = proxy_factory_fn(node, value)

        # Optionally set stack trace on the created Node for debugging purposes
        if fx_traceback.has_preserved_node_meta():
            current_meta: Dict[str, Any] = fx_traceback.get_current_meta()

            # Explicitly set the stack_trace, nn_module_stack and source_fn on the node.meta
            # If other meta fields are needed, they can be added here
            stack_trace = current_meta.get("stack_trace")
            if stack_trace:
                proxy.node.stack_trace = stack_trace

            nn_module_stack = current_meta.get("nn_module_stack")
            if nn_module_stack:
                proxy.node.meta["nn_module_stack"] = nn_module_stack

            source_fn = current_meta.get("source_fn")
            if source_fn:
                proxy.node.meta["source_fn"] = source_fn

        elif self.record_stack_traces:
            user_frame = self._find_user_frame()
            if user_frame:
                walk_stack_gen = traceback.walk_stack(user_frame)
                summary = traceback.StackSummary.extract(walk_stack_gen)  # type: ignore[arg-type]
                tb_lines = summary.format()
                proxy.node.stack_trace = ''.join(tb_lines)
        return proxy

    def call_module(
            self,
            m: torch.nn.Module,
            forward: Callable[..., Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            self.module_stack[_scope.module_path] = _scope.module_type
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                try:
                    self.concrete_mode = True
                    value = forward(*self.unpack_arg(args), **self.unpack_arg(kwargs))
                except UnsetValueException:
                    value = _UNSET
                finally:
                    self.concrete_mode = False
                ret_val = self.create_proxy(
                    'call_module', module_qualified_name, args, kwargs, value=value)
            key, _ = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f" Unexpected key {key}"
        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if ("proxy_factory_fn" in inspect.signature(self.create_proxy).parameters):
                            kwargs["proxy_factory_fn"] = (
                                None if not self.param_shapes_constant
                                # value is required by interface of proxy_factor_fn
                                else lambda node,
                                value: ParameterProxy(self, node, n, attr_val))
                        val_proxy = self.create_proxy(
                            "get_attr", n, (), {}, **kwargs, value=p)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args=None, value_args=None):
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError("``self`` argument cannot be part of *args expansion!")
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        sig = inspect.signature(fn_for_analysis)

        def proxy_placeholder(name: str):
            if value_args is not None and name in value_args:
                value = value_args[name]
            else:
                value = _UNSET

            if concrete_args is not None and name in concrete_args:

                cnt = 0

                def replace_ph(x):
                    nonlocal cnt
                    cnt += 1
                    param = sig.parameters[name]
                    default = (() if param.default is inspect.Parameter.empty else (param.default,))

                    out = self.create_proxy(
                        "placeholder", f"{name}_{str(cnt)}", default, {}, value=value)
                    if x == PH:
                        return out
                    # Union[int, bool] == bool in Python <= 3.6
                    if (type(x) == bool or type(x) in base_types and type(x) != torch.Tensor):
                        torch._assert(
                            out == x,
                            f"{name} has been specialized to have value {x} but got another value",
                        )
                    elif type(x) == type(None):
                        args = (
                            out,
                            f"{name} has been specialized to have value None but got another value",
                        )
                        self.create_proxy("call_function", _assert_is_none, args, {})
                    else:
                        warnings.warn(
                            f"Was not able to add assertion to guarantee correct input {name} to "
                            f"specialized function. It is up to the user to make sure that your inputs match the "
                            f"inputs you specialized the function with.")

                    return x

                return pytree.tree_map(replace_ph, concrete_args[name])
            if name[0] == "*":
                default = ()
            else:
                param = sig.parameters[name]
                default = () if param.default is inspect.Parameter.empty else (
                    param.default,)  # type: ignore[assignment]
            # Don't create a placeholder for values that default to None
            if value is _UNSET and default == (None,):
                return
            if value is _UNSET and default != ():
                value = param.default
            return self.create_proxy(
                "placeholder",
                name,
                default, {},
                type_expr=fn_for_analysis.__annotations__.get(name, None),
                value=value)

        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(
                    f"Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments"
                )
            concrete_args = {name: val for name, val in zip(arg_names, concrete_args)}
        args.extend(proxy_placeholder(names) for names in arg_names)

        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder("*" + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder("**" + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))

        flat_args, in_spec = pytree.tree_flatten(tuple(args))
        if any(not isinstance(i, pytree.LeafSpec) for i in in_spec.children_specs):
            # In the case that we have pytree-flattened inputs in
            # `concrete_args`, generate a flattening wrapper around the
            # original root function and return that.
            self.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(orig_args[:total_args], in_spec, None))

            def flatten_fn(*args):
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                out_args, out_spec = pytree.tree_flatten(tree_out)
                assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                self.graph._codegen.pytree_info = (
                    self.graph._codegen.pytree_info._replace(out_spec=out_spec))
                return out_args

            return flatten_fn, flat_args
        return root_fn, args

    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        value_args: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                self.root = root

                assert hasattr(
                    type(root), self.traced_func_name
                ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[Type["Tracer"]] = getattr(self, "__class__", None)
            self.graph = Graph(tracer_cls=tracer_cls)

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            assert isinstance(fn, FunctionType)

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args, value_args)

            parameter_proxy_cache: Dict[str, Proxy] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                if self.concrete_mode:
                    return attr_val
                else:
                    return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):

                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                if self.concrete_mode:
                    return mod(*args, **kwargs)
                else:
                    return self.call_module(mod, forward, args, kwargs)

            with _Patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
                # modified to propagate values
                _patch_wrapped_value_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(fn(*args)),),
                    {},
                    type_expr=fn.__annotations__.get("return", None),
                )

            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph


def _create_wrapped_value_func(orig_fn, visible_to_make_fx=False):

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Given an closed-over ``orig_function`` to invoke, search the args and kwargs for
        a Proxy object. If there is one, emit a ``call_function`` node to preserve the
        call to this leaf function directly. Otherwise, just return the results of
        this function call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            try:
                value = orig_fn(*proxy.tracer.unpack_arg(args), **proxy.tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return_proxy = proxy.tracer.create_proxy(
                "call_function", orig_fn, args, kwargs, value=value)
            return_proxy.node.meta["is_wrapped"] = True
            if visible_to_make_fx:
                return_proxy.node.meta["visible_to_make_fx"] = 1
            return return_proxy

        # Check if we want to trace this function for proxy tensors created via `make_fx`
        if visible_to_make_fx:
            # import here to avoid circular imports
            if torch_version < version.parse('2.1'):
                from brevitas.backport.fx.experimental.proxy_tensor import \
                    disable_proxy_modes_tracing
                from brevitas.backport.fx.experimental.proxy_tensor import get_innermost_proxy_mode
                from brevitas.backport.fx.experimental.proxy_tensor import proxy_call
            else:
                from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
                from torch.fx.experimental.proxy_tensor import get_innermost_proxy_mode
                from torch.fx.experimental.proxy_tensor import proxy_call

            # If there is no input with proxy, see if we are tracing with proxy tensors
            proxy_mode = get_innermost_proxy_mode()
            if proxy_mode is not None:
                # Disable tracing of the interior of the wrapped fn while evaluating
                with disable_proxy_modes_tracing():
                    out = proxy_call(proxy_mode, orig_fn, args, kwargs)
                return out

        return orig_fn(*args, **kwargs)

    return wrapped


def _create_wrapped_value_method(cls, name):
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Search the args and kwargs for a Proxy object. If there is one,
        emit a ``call_method`` node to preserve the call to this method
        directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            try:
                value = orig_fn(*proxy.tracer.unpack_arg(args), **proxy.tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return proxy.tracer.create_proxy('call_method', name, args, kwargs, value=value)
        else:
            value = orig_fn(*args, **kwargs)
        return value

    return wrapped


def _patch_wrapped_value_functions(patcher: _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for frame_dict, name, visible_to_make_fx in reversed(_wrapped_fns_to_patch):
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_value_func(orig_fn, visible_to_make_fx))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_value_method(cls, name))
