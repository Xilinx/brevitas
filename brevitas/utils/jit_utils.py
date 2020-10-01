import inspect
from mock import patch
from packaging import version

try:
    from torch._jit_internal import get_torchscript_modifier
except:
    get_torchscript_modifier = None

from torch.jit import script_method
import torch
from torch.version import __version__
from brevitas.inject import BaseInjector as Injector

IS_ABOVE_110 = version.parse(__version__) > version.parse('1.1.0')


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


def script_method_110_disabled(fn):
    if not IS_ABOVE_110:
        return fn
    else:
        return script_method(fn)