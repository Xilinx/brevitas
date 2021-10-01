"""
Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
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


from typing import Union, Callable, Optional, Dict, Any, List, Tuple
from types import FunctionType, ModuleType
import functools
import inspect
import builtins
import operator
import math

try:
    from torch._C import ScriptObject
except:
    ScriptObject = None

from torch.nn import Module

from brevitas.quant_tensor import QuantTensorBase
from . import _Patcher, _patch_function, _orig_module_call, _orig_module_getattr
from . import _find_proxy, _wrapped_methods_to_patch, _wrapped_fns_to_patch
from . import _autowrap_check
from . import *

_UNSET = object()
extended_base_types = base_types + (QuantTensorBase,)


class UnsetValueException(Exception):
    pass


class ValueProxy(Proxy):

    def __init__(self, node: Node, value, tracer = None):
        super(ValueProxy, self).__init__(node, tracer)
        if isinstance(value, Proxy):
            raise RuntimeError("Value of a proxy can't be a proxy.")
        self.value = value

    @property
    def value(self):
        if self._value is _UNSET:
            raise UnsetValueException
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __getattr__(self, k) -> 'ValueAttribute':
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        try:
            value = getattr(self.value, k)
        except UnsetValueException:
            value = _UNSET
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

    def __torch_function__(self, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        if is_tensor_method_or_property(orig_method):
            try:
                value = self.value(*self.tracer.unpack_arg(args), **self.tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return self.tracer.create_proxy(
                'call_method', orig_method.__name__, args, kwargs, value=value)
        else:
            try:
                value = orig_method(*self.tracer.unpack_arg(args), **self.tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            name = self.tracer.graph._target_to_str(orig_method.__name__)
            return self.tracer.create_proxy(
                'call_function', orig_method, args, kwargs, name, value=value)


class ValueAttribute(ValueProxy):
    def __init__(self, root: Proxy, attr: str, value: Any):
        self.root = root
        self.attr = attr
        self.value = value
        self.tracer = root.tracer
        self._node: Optional[Node] = None

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
    def scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            try:
                value = target(*tracer.unpack_arg(args), **tracer.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            return tracer.create_proxy('call_function', target, args, kwargs, value=value)
        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(ValueProxy, as_magic, impl)
    scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name}__'

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

    def __init__(self, autowrap_modules : Tuple[ModuleType] = (math, )):
        super(ValueTracer, self).__init__(autowrap_modules)
        self.concrete_mode = False

    def to_bool(self, obj: 'Proxy') -> bool:
        return obj.value.__bool__()

    def iter(self, obj: 'Proxy'):
        return self.create_proxy(
            'call_function', iter, (obj,), {}, value=obj.value.__iter__())

    # TODO deal with keys and values
    def keys(self, obj: 'Proxy') -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return ValueAttribute(obj, 'keys')()

    def proxy(self, node: Node, value: Any) -> 'Proxy':
        return ValueProxy(node, value, self)

    def unpack_arg(self, a: Any):
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.unpack_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore
        elif isinstance(a, (tuple, list)):
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

        if isinstance(a, Proxy):
            # base case: we unwrap the Proxy object
            return a.value
        elif isinstance(a, extended_base_types) or a is None or a is ...:
            return a

        raise NotImplementedError(f"argument of type: {type(a)}")

    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None,
                     type_expr : Optional[Any] = None, value: Any = _UNSET):
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
        return self.proxy(node, value)

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
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
        if not self.is_leaf_module(m, module_qualified_name):
            return forward(*args, **kwargs)
        else:
            try:
                self.concrete_mode = True
                value = forward(*self.unpack_arg(args), **self.unpack_arg(kwargs))
            except UnsetValueException:
                value = _UNSET
            finally:
                self.concrete_mode = False
            return self.create_proxy(
                'call_module', module_qualified_name, args, kwargs, value=value)

    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
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
        names_iter = iter(co.co_varnames)
        args : List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        def proxy_placeholder(name: str):
            if concrete_args is not None and name in concrete_args:
                value = concrete_args[name]
            else:
                param = inspect.signature(fn_for_analysis).parameters[name]
                value = _UNSET if param.default is inspect.Parameter.empty else param.default
            type_expr = fn_for_analysis.__annotations__.get(name, None)
            return self.create_proxy(
                'placeholder', name, (), {}, type_expr=type_expr, value=value)

        args.extend(proxy_placeholder(next(names_iter)) for _ in range(skip_arg_idx, total_args))

        # TODO values for *args and **kwargs
        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder('*' + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder('**' + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))

        return root_fn, args

    def trace(
            self,
            root: Union[Module, Callable],
            concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        if isinstance(root, Module):
            self.root = root
            fn = type(root).forward
        else:
            self.root = Module()
            fn = root
        self.graph = Graph()

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
        # in create_arg
        self.tensor_attrs: Dict[torch.Tensor, str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, tuple(i for i in [torch.Tensor, ScriptObject] if i is not None)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(
            fn, isinstance(root, torch.nn.Module), concrete_args)

        parameter_proxy_cache: Dict[str, Proxy] = {}  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly used.
        # Thus, we need to insert a proxy when __getattr__ requests a parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            if not self.concrete_mode and isinstance(attr_val, torch.nn.Parameter):
                for n, p in self.root.named_parameters():
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            parameter_proxy_cache[n] = self.create_proxy(
                                'get_attr', n, (), {}, value=p)
                        return parameter_proxy_cache[n]
            return attr_val

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(patcher, getattr(getattr(mod, "forward", mod), "__globals__", {}),
                            self._autowrap_function_ids)
            if self.concrete_mode:
                return forward(*args, **kwargs)
            else:
                return self.call_module(mod, forward, args, kwargs)

        with _Patcher() as patcher:
            # allow duplicate patches to support the case of nested calls
            patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper,
                                 deduplicate=False)
            patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper,
                                 deduplicate=False)
            _patch_wrapped_value_functions(patcher)
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)

            self.create_node('output', 'output', (self.create_arg(fn(*args)),), {})

        return self.graph


def _create_wrapped_value_func(orig_fn):
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
            return proxy.tracer.create_proxy('call_function', orig_fn, args, kwargs, value=value)
        else:
            value = orig_fn(*args, **kwargs)
        return value

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


def _patch_wrapped_value_functions(patcher : _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for frame_dict, name in _wrapped_fns_to_patch:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_value_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_value_method(cls, name))


