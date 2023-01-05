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


def clear_class_registry():
    # torch.jit.trace leaks memory, this should help
    # https://github.com/pytorch/pytorch/issues/86537
    # https://github.com/pytorch/pytorch/issues/35600
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._script_classes.clear()

