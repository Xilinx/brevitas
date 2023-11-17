"""
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

Forked from PyTorch 2.0.1 with import edits to point to backport
+ https://github.com/pytorch/pytorch/blob/7ac68cb648c1e8c5f53efe6696cb06d3c8e9853b/torch/fx/experimental/proxy_tensor.py
from https://github.com/pytorch/pytorch/pull/94461/files
"""

import contextlib
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
import functools
import inspect
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._python_dispatch import _pop_mode_temporarily
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree

from brevitas import backport
from brevitas.backport import fx
from brevitas.backport import SymBool
from brevitas.backport import SymFloat
from brevitas.backport import SymInt
from brevitas.backport.fx import GraphModule
from brevitas.backport.fx import Proxy
from brevitas.backport.fx import Tracer
from brevitas.backport.fx.passes.shape_prop import _extract_tensor_metadata
from brevitas.backport.utils._stats import count
from brevitas.backport.utils.weak import WeakTensorKeyDictionary

from .symbolic_shapes import ShapeEnv
from .symbolic_shapes import SymDispatchMode
from .symbolic_shapes import SymNode

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "py_sym_types",
    "get_innermost_proxy_mode"]
aten = torch.ops.aten
prim = torch.ops.prim

CURRENT_DECOMPOSITION_TABLE = {}

CONSTANT_NUMEL_LIMIT = 1

# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree._register_pytree_node(torch.Size, lambda x: (list(x), None), lambda xs, _: tuple(xs))


def fake_signature(fn, nargs):
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table


# ensure we cannot collide with other properties
proxy_slot = object()
no_default = object()

py_sym_types = (SymInt, SymFloat, SymBool)


def is_sym_node(node):
    assert hasattr(node, 'meta'), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta['val'], py_sym_types)


def set_proxy_slot(obj, tracer, proxy):
    if isinstance(obj, torch.Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        tracer.tensor_tracker[obj] = proxy
    else:
        # NB: Never clobber pre-existing proxy.  Although the proxies
        # are in principle equivalent, when we do graph partitioning
        # we need there not to be spurious dependencies on tangent inputs.
        # This works because primals get their SymInts set first, and
        # THEN later we allocate tangent inputs.  Make sure if a SymInt
        # is derivable from a primal that we use that.
        assert isinstance(obj, SymNode), type(obj)
        if obj not in tracer.symnode_tracker:
            tracer.symnode_tracker[obj] = proxy


def has_proxy_slot(obj, tracer):
    assert isinstance(obj, (torch.Tensor, SymNode)), type(obj)
    return get_proxy_slot(obj, tracer, False, lambda _: True)


# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(obj, tracer, default=no_default, transform=lambda x: x):
    if isinstance(obj, torch.Tensor):
        tracker = tracer.tensor_tracker
    else:
        assert isinstance(obj, SymNode), type(obj)
        tracker = tracer.symnode_tracker

    if obj not in tracker:
        if default is no_default:
            raise RuntimeError(f"{obj} is not tracked with proxy for {tracer}")
        return default
    return transform(tracker[obj])


def snapshot_fake(val):
    return val.detach()


def unwrap_proxy(proxy_mode, e):
    if isinstance(e, torch.Tensor):
        return get_proxy_slot(e, proxy_mode.tracer, e, lambda e: e.proxy)
    elif isinstance(e, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return get_proxy_slot(e.node, proxy_mode.tracer, e, lambda e: e())
    else:
        return e


# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy, val):
    if isinstance(val, FakeTensor):
        proxy.node.meta['val'] = snapshot_fake(val)
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    elif isinstance(val, py_sym_types):
        proxy.node.meta['val'] = val
    elif isinstance(val, (list, tuple)):
        if all(isinstance(x, FakeTensor) for x in val):
            proxy.node.meta['val'] = [snapshot_fake(x) for x in val]
    elif isinstance(val, torch.Tensor):
        if not val.is_sparse:
            proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                proxy.node.meta['val'] = torch.empty_strided(
                    val.shape, val.stride(), device=val.device, dtype=val.dtype)
    return proxy


def thunkify(f, *args, **kwargs):
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    return functools.lru_cache(1)(functools.partial(f, *args, **kwargs))


def track_tensor(tensor, proxy, *, constant, tracer):

    def try_set_proxy_slot(outer_s, proxy_callable, *args):
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            inner_s = outer_s.node
            set_proxy_slot(inner_s, tracer, thunkify(proxy_callable, outer_s, *args))

    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_size(proxy, i), x), i)

    for i, s in enumerate(tensor.stride()):
        try_set_proxy_slot(s, lambda x, i: set_meta(torch.ops.aten.sym_stride(proxy, i), x), i)

    try_set_proxy_slot(tensor.numel(), lambda x: set_meta(torch.ops.aten.sym_numel(proxy), x))
    try_set_proxy_slot(
        tensor.storage_offset(), lambda x: set_meta(torch.ops.aten.sym_storage_offset(proxy), x))
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))


def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):

    def wrap_with_proxy(e, proxy, constant):
        if isinstance(e, torch.Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e.node, tracer, lambda: proxy)
        elif isinstance(e, list):
            # example use case: allreduce_ returns ([tensor], work)
            for idx, ee in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], get_constant(idx))

    def get_constant(idx):
        if constant is None:
            return None
        else:
            return constant[idx]

    # Unfortunately, tree_map cannot directly be used here. As the resulting
    # object may be a proxy that represents a tuple, we may need to
    # explicitly unwrap the proxy by simulating the flattening operations.
    if isinstance(inner_res, (tuple, list)):
        if isinstance(proxy_res, fx.Proxy):
            set_meta(proxy_res, inner_res)
        for idx, e in enumerate(inner_res):
            wrap_with_proxy(e, proxy_res[idx], get_constant(idx))
    elif isinstance(inner_res, py_sym_types + (torch.Tensor,)):
        wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


def maybe_disable_fake_tensor_mode():
    # TODO: figure out if this API generally makes sense and bake it into the
    # library
    mb_fake_mode = _get_current_dispatch_mode()
    if isinstance(mb_fake_mode, FakeTensorMode):
        return _pop_mode_temporarily()
    else:
        return nullcontext()


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[torch.Tensor]


def fetch_sym_proxy(tracer):

    def inner(e):
        n = e.node
        if n.constant is not None:
            return n.constant
        else:
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(n, tracer)()

    return inner


def fetch_tensor_proxy(tracer):
    return lambda t: get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (torch.Tensor, torch.nn.Parameter)


def proxy_call(proxy_mode, func, args, kwargs):
    # `__torch_dispatch__` is only called on torch ops, which must subclass `OpOverload`
    # We treat all other functions as an `external_call`, for instance, a function decorated
    # with `@torch.fx.wrap`
    external_call = not isinstance(func, (backport._ops.OpOverload, torch._ops.OpOverload))

    def can_handle_tensor(x):
        return type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not pytree.tree_all_only(torch.Tensor, can_handle_tensor, (args, kwargs)):
        return NotImplemented

    if not external_call:
        if func in CURRENT_DECOMPOSITION_TABLE:
            with proxy_mode:
                r = CURRENT_DECOMPOSITION_TABLE[func](*args, **kwargs)
                if r is not NotImplemented:
                    return r
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    tracer = proxy_mode.tracer
    f_args, f_kwargs = pytree.tree_map_only(torch.Tensor, fetch_tensor_proxy(tracer), (args, kwargs))

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        pytree.tree_all_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and pytree.tree_all_only((SymInt, SymFloat, SymBool), lambda _: False, (args, kwargs)))
    if not external_call and torch.Tag.data_dependent_output in func.tags:  # type: ignore[attr-defined]
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_args, const_kwargs = pytree.tree_map_only(
                _ProxyTensor, lambda t: t.constant, (f_args, f_kwargs)
            )
            with maybe_disable_fake_tensor_mode():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if pytree.tree_all_only(torch.Tensor,
                                lambda t: not isinstance(t, FakeTensor), (args, kwargs)):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call.")
    proxy_args, proxy_kwargs = pytree.tree_map_only(
        (SymInt, SymFloat, SymBool),
        fetch_sym_proxy(proxy_mode.tracer),
        pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (f_args, f_kwargs))
    )

    # When we trace through a torch.tensor invocation, you never actually
    # see a torch.ops.aten.tensor call. Instead, the way this function is
    # implemented internally is that we allocate a plain tensor (this is
    # *guaranteed* to be a plain tensor, we disable all modes when doing
    # so), and then call at::lift_fresh on it (to give modes a chance to do
    # their stuff).  Furthermore, the tensor argument to lift_fresh is guaranteed
    # to be freshly allocated, so we want lift_fresh to be a no-op (directly
    # returning the input argument).
    #
    # Here is the basic problem: when we trace this sequence of executions
    # into an FX graph, what happens to this call sequence?  Traditionally,
    # tensor constants get interned as buffers on the FX GraphModule.  But
    # this is dangerous.  Consider:
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default

    if external_call:
        proxy_out = proxy_mode.tracer.create_proxy(
            'call_function', func, proxy_args, proxy_kwargs, name=func.__name__)
    else:
        proxy_out = proxy_mode.tracer.create_proxy(
            'call_function',
            func,
            proxy_args,
            proxy_kwargs,
            name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__))

        # This makes DCE marginally less likely to DCE inplace operations.
        # It is not strictly necessary
        # Kind of a hacky way to test if an op is in-place or not
        if func.overloadpacket.__name__[-1] == "_" and func.overloadpacket.__name__[0] != "_":
            if isinstance(args[0], List):
                # e.g., c10d::allreduce_ returns a list of tensors as the first element
                # in the output.
                for i, a in enumerate(args[0]):
                    a.proxy = proxy_out[0][i]
            else:
                args[0].proxy = proxy_out

    out = func(*args, **kwargs)

    # In some circumstances, we will be tracing in a situation where a tensor
    # is *statically* known to be a constant (currently, this only happens if
    # you run torch.tensor; deterministic factory functions like torch.arange
    # don't get this treatment).  When the tensor in question is small, it's
    # helpful to due constant propagation in case we call item() (in which
    # case we can return the constant value that is known, rather than give
    # an error.)  The logic here tests if constant propagation is possible
    # (because all of the inputs are constant).  If so, we disable fake tensor
    # mode (if it is on) and do true compute on the constant.
    #
    # It's worth highlighting that we're making a policy decision here.
    # There is a potential that the tensor is actually quite large, and we
    # don't actually want to run the compute.  The tensor being quite large
    # is one of the reasons why factory functions don't get this treatment
    # (since they can be quite large; if a parameter is initialized to a
    # constant value it will be!)  Similarly, there is also a potential
    # to run an operator that blows up the size of a small tensor; we don't
    # protect against this case, but we could force, e.g., only single
    # element constant computation by testing the numel of the result before
    # propagating const-ness.  Similarly, we don't require the constant to
    # live on CPU, but we could.
    any_constant = pytree.tree_any_only(
        _ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))

    constant = None

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if func is torch.ops.aten.lift_fresh_copy.default and out.numel() <= CONSTANT_NUMEL_LIMIT:
        with maybe_disable_fake_tensor_mode():
            constant = args[0].clone()
    elif ((external_call or torch.Tag.nondeterministic_seeded not in func.tags
          )  # type: ignore[attr-defined]
          and all_constant and any_constant and
          pytree.tree_all_only(torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out)):
        # NB: do NOT include factories as constants
        with maybe_disable_fake_tensor_mode():
            const_args, const_kwargs = pytree.tree_map_only(
                _ProxyTensor, lambda t: t.constant, (f_args, f_kwargs)
            )
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None

    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    return out


class PythonKeyTracer(Tracer):

    def __init__(self):
        super().__init__(autowrap_modules=())
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = weakref.WeakKeyDictionary()  # type: ignore[var-annotated]

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
            self,
            m: torch.nn.Module,
            forward: Callable[..., Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(self, attr, attr_val, parameter_proxy_cache):
        return attr_val

    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                i = 0
                while True:
                    qualname = f'_param_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        elif isinstance(a, (SymInt, SymFloat, SymBool)):
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)


def dispatch_trace(
    root: Union[torch.nn.Module, Callable],
    tracer: Tracer,
    concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


def wrap_key(f, tensors, tracer):
    flat_tensors, tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies):
        flat_proxies, proxies_spec = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        assert isinstance(_get_current_dispatch_mode(), ProxyTorchDispatchMode)
        with _pop_mode_temporarily():
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        out = f(*tensors)
        out = pytree.tree_map_only(
            torch.Tensor, lambda t: get_proxy_slot(t, tracer, t, lambda x: x.proxy), out)
        out = pytree.tree_map_only((SymInt, SymFloat, SymBool),
                                   lambda t: get_proxy_slot(t.node, tracer)(),
                                   out)
        return out

    return wrapped


class ProxyTorchDispatchMode(TorchDispatchMode):

    def __init__(self, tracer, tracing_mode):
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.enable_tracing = True
        self.sym_mode = ProxySymDispatchMode(tracer)
        self.trace_state = {}
        self._managers = []

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with self.sym_mode.enable(False):
            return self.inner_torch_dispatch(func, types, args, kwargs)

    def __enter__(self):
        # sym mode first, then us...
        m = self.sym_mode.enable(True)
        self._managers.append(m)
        m.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        m = self._managers.pop()
        # ...exit us first, then sym mode
        b = super().__exit__(exc_type, exc_value, traceback)
        if not b:
            return m.__exit__(exc_type, exc_value, traceback)
        else:
            return m.__exit__(None, None, None)

    def inner_torch_dispatch(self, func, types, args=(), kwargs=None):
        if not self.enable_tracing:
            return func(*args, **kwargs)

        if func in [prim.device.default]:
            return func(*args, **kwargs)

        out = proxy_call(self, func, args, kwargs)
        return out


class ProxySymDispatchMode(SymDispatchMode):

    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        # When false, we don't trace operations.  If you do this, you MUST
        # call track_tensor/track_tensor_tree on all results of the operation
        # to ensure we can adeduately track the results
        self.enable_tracing = True

    @contextmanager
    def enable(self, b):
        old = self.enable_tracing
        self.enable_tracing = b
        try:
            yield
        finally:
            self.enable_tracing = old

    def _compute_proxy(self, func, args, out: Union[SymInt, SymFloat, SymBool]):
        n_args = tuple(
            get_proxy_slot(a.node, self.tracer)().node if isinstance(a, py_sym_types) else a
            for a in args)

        # func doesn't have a __torch_function__ that Proxy can interpose, so
        # we gotta do it manually
        n_out = self.tracer.create_node("call_function", func, n_args, {})
        p_out = fx.Proxy(n_out, self.tracer)
        set_meta(p_out, out)
        return p_out

    def __sym_dispatch__(self, func, types, args, kwargs):
        if not self.enable_tracing:
            return func(*args, **kwargs)

        # Peephole optimize multiply by one
        # NB: be careful not to trigger guards here!
        if func == operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # For speed, we assume there are no nested data structures
        # (otherwise we could use tree_map)
        # We also assume there are no keyword arguments.
        assert not kwargs
        out = func(*args, **kwargs)
        assert isinstance(out, py_sym_types), f"{func}(*{args}, **{kwargs}) = {out}"

        # Delays tracing out the proxies on this op until we actually need it
        p_out_thunk = thunkify(self._compute_proxy, func=func, args=args, out=out)
        set_proxy_slot(out.node, self.tracer, p_out_thunk)
        return out


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(torch.fx.Interpreter):

    def __init__(
            self,
            module: torch.fx.GraphModule,
            new_graph: torch.fx.Graph,
            decomposition_table=None,
            **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        # Blegh
        self.tracer.tensor_tracker = WeakTensorKeyDictionary()  # type: ignore[attr-defined]
        self.tracer.symnode_tracker = weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode="real")

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # TODO handle case where the first character of target is '*'
        return out

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)

        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        # Should enter the mode at least once for being able to restore it later
        # See: https://github.com/pytorch/pytorch/pull/82549#discussion_r934782025
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)


def wrapper_and_args_for_make_fx(func, args, kwargs):
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args):
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)

    return wrapped, flat_args


@contextmanager
def disable_autocast_cache():
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


def make_fx(f, decomposition_table=None, tracing_mode="real", _allow_non_fake_inputs=False):
    assert tracing_mode in ["real", "fake", "symbolic"]

    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda _: fx.PH, args)  # type: ignore[attr-defined]
        fx_tracer = PythonKeyTracer()
        fake_tensor_mode: Any = nullcontext()
        if tracing_mode == "real":
            fake_tensor_mode = nullcontext()
        elif tracing_mode == "fake":
            fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=True, allow_non_fake_inputs=_allow_non_fake_inputs)
        elif tracing_mode == "symbolic":
            shape_env = ShapeEnv()
            fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=_allow_non_fake_inputs,
                shape_env=shape_env)
        else:
            raise AssertionError(f"Unexpected tracing type: {tracing_mode}")

        python_dispatcher_mode: Any = nullcontext()
        if tracing_mode == "symbolic":
            python_dispatcher_mode = enable_python_dispatcher()

        proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode)

        arg_count = 0

        def wrap_fake(x):
            nonlocal arg_count
            if isinstance(x, torch.Tensor):
                # TODO: it would be nice to line these up with the names
                # FX will choose for the placeholders, but we don't
                # actually know what the names will be at this point yet
                # NB: the Source here is actually meaningless
                from torch._dynamo.source import ConstantSource
                source = ConstantSource(f"input{arg_count}")
                arg_count += 1
                return fake_tensor_mode.from_tensor(x, source=source)  # type: ignore[attr-defined]

            return x

        sym_mode = proxy_mode.sym_mode

        wrap_fn_map = {
            "real": lambda x: x,
            "fake": wrap_fake,
            "symbolic": wrap_fake,}
        args = pytree.tree_map(wrap_fn_map[tracing_mode], args)

        if not hasattr(inspect.unwrap(f),
                       '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
            # FX doesn't support varargs, so we gotta fake up a wrapper
            # TODO: Would be nice to fix this at the source...
            func = fake_signature(f, len(phs))
        else:
            func = f

        # We disable the autocast cache as the autocast cache causes type conversions on parameters to
        # check a cache, which introduces untracked tensors into the graph
        #
        # We also disable tracing by any other tensor proxy-based tracers except the current. The
        # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
        # thus irrelevant to any external functional trace.
        with decompose(decomposition_table), fake_tensor_mode, python_dispatcher_mode, \
             sym_mode, proxy_mode, disable_autocast_cache(), disable_proxy_modes_tracing(enable_current=True):
            t = dispatch_trace(
                wrap_key(func, args, fx_tracer), tracer=fx_tracer, concrete_args=tuple(phs))

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if tracing_mode == "symbolic":
            t.shape_env = shape_env  # type: ignore[assignment]
        return t

    return wrapped


def get_torch_dispatch_modes():
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()


def get_innermost_proxy_mode():
    for m in reversed(torch.utils._python_dispatch._get_current_dispatch_mode_stack()):
        if isinstance(m, ProxyTorchDispatchMode):
            return m
    return None


@contextlib.contextmanager
def disable_proxy_modes_tracing(enable_current=False):
    modes = get_torch_dispatch_modes()
    proxy_tensor_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
    if enable_current:
        proxy_tensor_modes = proxy_tensor_modes[:-1]
    olds = [(m.enable_tracing, m.sym_mode.enable_tracing) for m in proxy_tensor_modes]
    for proxy_mode in proxy_tensor_modes:
        proxy_mode.enable_tracing = False
        proxy_mode.sym_mode.enable_tracing = False
    try:
        yield
    finally:
        for proxy_mode, (old, old_sym) in zip(proxy_tensor_modes, olds):
            proxy_mode.enable_tracing = old
            proxy_mode.sym_mode.enable_tracing = old_sym


def get_isolated_graphmodule(func, args, kwargs, tracing_mode="real"):
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(wrapped, tracing_mode=tracing_mode)(all_args)
    return gm
