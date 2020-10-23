from unittest import mock
import inspect

import torch

from .backport.signatures import get_torch_overrides
from .backport.signatures import get_nn_functional_overrides
from .backport.signatures import get_testing_overrides
from .backport._overrides import torch_function_dispatch
from .backport._overrides import _implement_torch_function


def make_above_16_patches():

    original_torch_cat = torch.cat
    original_torch_stack = torch.stack

    def above_16_cat(tensors, dim, out=None):
        if not isinstance(tensors, (tuple, list)):
            tensors = tuple(*tensors)
        if out is not None:
            return original_torch_cat(tensors, dim, out)
        else:
            return original_torch_cat(tensors, dim)

    def above_16_stack(tensors, dim, out=None):
        if not isinstance(tensors, (tuple, list)):
            tensors = tuple(*tensors)
        if out is not None:
            return original_torch_stack(tensors, dim, out)
        else:
            return original_torch_stack(tensors, dim)

    above_16_cat_patch = mock.patch('torch.cat', wraps=above_16_cat)
    above_16_stack_patch = mock.patch('torch.stack', wraps=above_16_stack)

    return [above_16_cat_patch, above_16_stack_patch]


def make_equal_16_patches():

    original_torch_cat = torch.cat
    original_torch_stack = torch.stack

    def equal_16_cat(tensors, dim, out=None):
        if isinstance(tensors, (tuple, list)):
            kwargs = {'tensors': tensors, 'dim': dim}
            if out is not None:
                kwargs['out'] = out
            return _implement_torch_function(original_torch_cat, tensors, [], kwargs)
        else:
            tensors = tuple(t for t in tensors)
            if out is not None:
                return equal_16_cat(tensors, dim, out)
            else:
                return equal_16_cat(tensors, dim)

    def equal_16_stack(tensors, dim, out=None):
        if isinstance(tensors, (tuple, list)):
            kwargs = {'tensors': tensors, 'dim': dim}
            if out is not None:
                kwargs['out'] = out
            return _implement_torch_function(original_torch_stack, tensors, [], kwargs)
        else:
            tensors = tuple(t for t in tensors)
            if out is not None:
                return equal_16_stack(tensors, dim, out)
            else:
                return equal_16_stack(tensors, dim)

    equal_16_cat_patch = mock.patch('torch.cat', wraps=equal_16_cat)
    equal_16_stack_patch = mock.patch('torch.stack', wraps=equal_16_stack)

    return [equal_16_cat_patch, equal_16_stack_patch]


def make_below_16_patches():
    EXCLUDED_TORCH = [torch.cat, torch.stack]

    def make_dispatcher(fn):
        lambda_signature = get_testing_overrides()[fn]
        signature = inspect.signature(lambda_signature)
        param_to_dispatch = lambda p: p.default == None or p.default == inspect.Parameter.empty
        params_to_dispatch = [n for n, p in signature.parameters.items() if param_to_dispatch(p)]
        args = str(signature)[1:-1]  # remove ()
        returns = ','.join(params_to_dispatch)
        dispatcher_source = f"lambda {args}: ({returns},)"  # force tuple
        dispatcher = eval(dispatcher_source)
        return dispatcher

    torch_p = 'torch.'
    func_p = 'torch.nn.functional.'
    dispatch = lambda fn: torch_function_dispatch(make_dispatcher(fn))(fn)
    torch_to_override = [fn for fn in get_torch_overrides().keys() if fn not in EXCLUDED_TORCH]
    func_to_override = get_nn_functional_overrides().keys()
    torch_override = {fn: dispatch(fn) for fn in torch_to_override}
    func_override = {fn: dispatch(fn) for fn in func_to_override}
    make_patch = lambda prefix, fn, wrapper: mock.patch(prefix + fn.__name__, wraps=wrapper)
    torch_patches = [make_patch(torch_p, fn, wrap) for fn, wrap in torch_override.items()]
    func_patches = [make_patch(func_p, fn, wrap) for fn, wrap in func_override.items()]
    return torch_patches + func_patches


ABOVE_16_PATCHES = make_above_16_patches()
EQUAL_16_PATCHES = make_equal_16_patches()
BELOW_16_PATCHES = make_equal_16_patches() + make_below_16_patches()
