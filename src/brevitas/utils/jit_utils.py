import inspect
from unittest.mock import patch

try:
    from torch._jit_internal import get_torchscript_modifier
except:
    get_torchscript_modifier = None

from torch.jit import script_method
import torch

from brevitas.inject import BaseInjector as Injector
from brevitas.jit import IS_ABOVE_110


def _get_modifier_wrapper(fn):
    if inspect.isclass(fn) and issubclass(fn, Injector):
        return None
    else:
        return get_torchscript_modifier(fn)


def jit_trace_patched(*args, **kwargs):
    if IS_ABOVE_110:
        with patch('torch._jit_internal.get_torchscript_modifier', wraps=_get_modifier_wrapper):
            return torch.jit.trace(*args, **kwargs)
    else:
        return torch.jit.trace(*args, **kwargs)


