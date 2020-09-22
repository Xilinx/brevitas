import inspect
from mock import patch
from packaging import version

try:
    from torch._jit_internal import get_torchscript_modifier
except:
    get_torchscript_modifier = None
import torch
from torch.version import __version__
from dependencies import Injector

PYTORCH_VERSION_THRESHOLD = version.parse('1.1.0')


def _get_modifier_wrapper(fn):
    if inspect.isclass(fn) and issubclass(fn, Injector):
        return None
    else:
        return get_torchscript_modifier(fn)


def jit_trace_patched(*args, **kwargs):
    if version.parse(__version__) > PYTORCH_VERSION_THRESHOLD:
        with patch('torch._jit_internal.get_torchscript_modifier', wraps=_get_modifier_wrapper):
            return torch.jit.trace(*args, **kwargs)
    else:
        return torch.jit.trace(*args, **kwargs)