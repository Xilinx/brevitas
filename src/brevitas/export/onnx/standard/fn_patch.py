from functools import partial
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from brevitas import config
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.mixin.base import _CachedIO
from brevitas.utils.python_utils import patch

from .function import QuantizeLinearFunction, DequantizeLinearFunction
from .handler.base import StdONNXQuantLayerHandler


_FN_TO_CACHE = [
    F.relu,
    F.relu6,
    F.hardtanh,
    F.max_pool1d,
    F.max_pool2d,
    F.max_pool3d,
    F.adaptive_max_pool1d,
    F.adaptive_max_pool2d,
    F.adaptive_max_pool3d,
]


_fn_cache = []


def gen_restore_fn_patches():
    return [patch(torch.nn.functional, fn.__name__, fn) for fn in _FN_TO_CACHE]


def _cache_fn_dispatcher(fn, input, *args, **kwargs):
    with ExitStack() as stack:
        # disable recursing into this patch
        for mgr in gen_restore_fn_patches():
            stack.enter_context(mgr)
        if isinstance(input, QuantTensor):
            inp_cache = None
            out_cache = None
            if input.is_not_none:
                inp_cache = _CachedIO(input, metadata_only=True)
            output = fn(input, *args, **kwargs)
            if isinstance(output, QuantTensor) and output.is_not_none:
                out_cache = _CachedIO(output, metadata_only=True)
            _fn_cache.append((inp_cache, out_cache))
        else:
            # could be a fn invoked within a quant module on a dequant tensor
            # or a function invoked on a float tensor. The former won't show
            # up during jit tracing as they are replaced by symbolic functions,
            # but the latter will, so we have to account for them in the _fn_cache
            output = fn(input, *args, **kwargs)
            if not config._IS_INSIDE_QUANT_LAYER:
                _fn_cache.append(None)
    return output


def _trace_fn_dispatcher(fn, input, *args, **kwargs):
    cached_io = _fn_cache.pop(0)
    if cached_io is not None:
        cached_inp, cached_out = cached_io
        if cached_inp is not None:
            deq_kwargs = StdONNXQuantLayerHandler.dequant_symbolic_kwargs_from_cached_io(cached_inp)
            input = DequantizeLinearFunction.apply(input, *deq_kwargs.values())
        output = fn(input, *args, **kwargs)
        if cached_out is not None:
            q_kwargs = StdONNXQuantLayerHandler.quant_symbolic_kwargs_from_cached_io(cached_out)
            output = QuantizeLinearFunction.apply(output, *q_kwargs.values())
    else:
        output = fn(input, *args, **kwargs)
    return output


def _gen_patches(fn_dispatcher):
    patches = []
    for fn in _FN_TO_CACHE:
        dispatcher = partial(fn_dispatcher, fn)
        p = patch(torch.nn.functional, fn.__name__, dispatcher)
        patches.append(p)
    return patches


def gen_cache_patches():
    return _gen_patches(_cache_fn_dispatcher)


def gen_trace_patches():
    return _gen_patches(_trace_fn_dispatcher)

