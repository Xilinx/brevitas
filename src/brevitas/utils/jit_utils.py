import inspect

import torch
try:
    from torch._jit_internal import get_torchscript_modifier
except:
    get_torchscript_modifier = None


from dependencies import Injector
from brevitas.inject import ExtendedInjector
from brevitas.jit import IS_ABOVE_110

from .python_utils import patch


def _get_modifier_wrapper(fn):
    if inspect.isclass(fn) and issubclass(fn, (Injector, ExtendedInjector)):
        return None
    else:
        return get_torchscript_modifier(fn)


if IS_ABOVE_110:
    def jit_patches_generator():
        return [patch(torch._jit_internal, 'get_torchscript_modifier', _get_modifier_wrapper)]
else:
    jit_patches_generator = None

