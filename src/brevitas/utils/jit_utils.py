import inspect
from unittest.mock import patch
from functools import partial

try:
    from torch._jit_internal import get_torchscript_modifier
except:
    get_torchscript_modifier = None

import torch

from dependencies import Injector
from brevitas.inject import ExtendedInjector
from brevitas.jit import IS_ABOVE_110


def _get_modifier_wrapper(fn):
    if inspect.isclass(fn) and issubclass(fn, (Injector, ExtendedInjector)):
        return None
    else:
        return get_torchscript_modifier(fn)


def _fn_patched(fn, *args, **kwargs):
    if IS_ABOVE_110:
        with patch('torch._jit_internal.get_torchscript_modifier', wraps=_get_modifier_wrapper):
            return fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


jit_trace_patched = partial(_fn_patched, torch.jit.trace)
onnx_export_patched = partial(_fn_patched, torch.onnx.export)

