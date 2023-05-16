import functools
import inspect

from torch import fx
from torch.fx import wrap
import torch.fx.experimental.proxy_tensor
from torch.fx.experimental.proxy_tensor import Any
from torch.fx.experimental.proxy_tensor import decompose
from torch.fx.experimental.proxy_tensor import disable_autocast_cache
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.experimental.proxy_tensor import dispatch_trace
from torch.fx.experimental.proxy_tensor import enable_python_dispatcher
from torch.fx.experimental.proxy_tensor import fake_signature
from torch.fx.experimental.proxy_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import nullcontext
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.experimental.proxy_tensor import pytree
from torch.fx.experimental.proxy_tensor import ShapeEnv
import torch.utils.cpp_extension


def wrappable_make_fx(
        f, decomposition_table=None, tracing_mode="real", _allow_non_fake_inputs=False):
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
                allow_fallback_kernels=True,
                allow_non_fake_inputs=_allow_non_fake_inputs,
            )
        elif tracing_mode == "symbolic":
            shape_env = ShapeEnv()
            fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=_allow_non_fake_inputs,
                shape_env=shape_env,
            )
        else:
            raise AssertionError(f"Unexpected tracing type: {tracing_mode}")

        python_dispatcher_mode: Any = nullcontext()
        if tracing_mode == "symbolic":
            python_dispatcher_mode = enable_python_dispatcher()

        proxy_mode = ProxyTorchDispatchMode(fx_tracer, tracing_mode)
        sym_mode = proxy_mode.sym_mode

        if (not hasattr(inspect.unwrap(f), "__code__") or
                inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS):
            func = fake_signature(f, len(phs))
        else:
            func = f

        with decompose(decomposition_table), fake_tensor_mode, python_dispatcher_mode, \
            sym_mode, proxy_mode, disable_autocast_cache(), disable_proxy_modes_tracing(enable_current=True):
            # Original:
            #   t = dispatch_trace(wrap_key(func, args, fx_tracer), tracer=fx_tracer, concrete_args=tuple(phs))
            t = dispatch_trace(func, tracer=fx_tracer, concrete_args=tuple(phs))

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if tracing_mode == "symbolic":
            t.shape_env = shape_env  # type: ignore[assignment]
        return t

    return wrapped
