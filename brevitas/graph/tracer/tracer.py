from contextlib import ExitStack
from _collections_abc import Iterable
from inspect import getcallargs
from typing import List, Union, Any
from copy import copy
import functools
from packaging import version
from dataclasses import field, dataclass, replace

import torch
from torch import Tensor

from brevitas.quant_tensor import QuantTensor
from .trace import Trace, TraceElem
from .wrapper.scriptmodule import torchscript_wrapper
from .wrapper.builtin import IntWrapper, StrWrapper, FloatWrapper
from ..module import FnType
from ..utils import flatten

if version.parse(torch.__version__) > version.parse('1.6.0'):
    from torch.overrides import get_testing_overrides
    from .patch import ABOVE_16_PATCHES as PATCHES
elif version.parse(torch.__version__) == version.parse('1.6.0'):
    from torch._overrides import get_testing_overrides
    from .patch import EQUAL_16_PATCHES as PATCHES
else:
    from .backport.signatures import get_testing_overrides
    from .patch import BELOW_16_PATCHES as PATCHES

TORCH_FN_NAMES = [fn.__name__ for fn in get_testing_overrides().keys()]
TORCH_FN_OVERRIDE_DICT = get_testing_overrides()

TYPES_TO_TRACE = [
    IntWrapper,
    FloatWrapper,
    StrWrapper,
    Iterable,  # works with tuple_iterator and such
    list,
    dict,
    tuple,
    torch.Tensor,
    QuantTensor,
    torch.Size]


class CallableWrapper(object):

    def __init__(self, tracer, callabl, inplace):
        self.tracer = tracer
        self.callabl = callabl
        self.inplace = inplace

    def __call__(self, *args, **kwargs):
        args, kwargs = self.tracer.repack_args_kwargs(args, kwargs)
        out = self.callabl(*args, **kwargs)
        kwargs['self'] = self.tracer.value_  # add tensor to kwargs
        out = self.tracer.update_trace(self.callabl.__name__, FnType.METHOD, args, kwargs, out)
        return self.tracer.epilogue(self.inplace, out)


# Adapted from: https://bit.ly/3hYCpvJ (stackoverflow)
class TracerMeta(type):

    # Adapted from: https://code.activestate.com/recipes/496741-object-proxying/
    # __next__ and __iter__ are removed since iterables are always unrolled
    magic_methods = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', '__contains__',
        '__delitem__', '__delslice__', '__div__', '__divmod__', '__eq__', '__float__',
        '__floordiv__', '__ge__', '__getitem__', '__getslice__', '__gt__', '__hash__',
        '__hex__', '__iadd__', '__iand__', '__idiv__', '__idivmod__', '__ifloordiv__',
        '__ilshift__', '__imod__', '__imul__', '__int__', '__invert__', '__ior__', '__ipow__',
        '__irshift__', '__isub__', '__itruediv__', '__ixor__', '__le__', '__len__',
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__oct__',
        '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__rdivmod__',
        '__reduce__', '__reduce_ex__', '__reversed__', '__rfloorfiv__',  '__rlshift__', '__rmod__',
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__',
        '__rxor__', '__setitem__', '__setslice__', '__sub__', '__truediv__', '__xor__',
        '__next__', '__iter__']

    @staticmethod
    def _magic_function(tracer, method_name, *args, **kwargs):
        if hasattr(tracer.value_, method_name) and method_name in TracerMeta.magic_methods:
            fn = getattr(tracer.value_, method_name)
            args, kwargs = tracer.repack_args_kwargs(args, kwargs)
            out = fn(*args, **kwargs)
            kwargs['self'] = tracer.value_
            out = tracer.update_trace(method_name, FnType.METHOD, args, kwargs, out)
            return tracer.return_output(out)

    def __new__(cls, name, bases, attr):
        new = super(TracerMeta, cls).__new__(cls, name, bases, attr)
        for method_name in TracerMeta.magic_methods:
            magic_method = functools.partialmethod(TracerMeta._magic_function, method_name)
            setattr(new, method_name, magic_method)
        return new


@dataclass
class Tracer(metaclass=TracerMeta):

    value_: Any
    trace_: Trace = field(default_factory=Trace)
    namespace_: List = field(default_factory=list)

    def __bool__(self):
        return self.value_

    def __getattr__(self, name: str):
        inplace_tensor = name.endswith('_') and name[:-1] in TORCH_FN_NAMES
        attr = getattr(self.value_, name)
        if callable(attr):
            return CallableWrapper(self, attr, inplace_tensor)
        else:
            kwargs = {'self': self.value_}
            self.update_trace(name, FnType.ATTRIBUTE, [], kwargs, attr)
            return self.return_output(attr)

    def repack_value(self, value):
        if self.trace_.index_from_map(value) is None:
            if isinstance(value, Tracer):
                return value.value_
            elif type(value) is int:
                return IntWrapper(value)
            elif type(value) is float:
                return FloatWrapper(value)
            elif type(value) is str:
                return StrWrapper(value)
            elif isinstance(value, tuple) and not isinstance(value, QuantTensor):
                return tuple(self.repack_value(v) for v in value)
            elif isinstance(value, list):
                return [self.repack_value(v) for v in value]
            elif isinstance(value, dict):
                return {self.repack_value(k): self.repack_value(v) for k, v in value.items()}
            else:
                return value
        else:
            return value

    def repack_args_kwargs(self, args, kwargs):
        new_args = [self.repack_value(a) for a in args]
        new_kwargs = {k: self.repack_value(a) for k, a in kwargs.items()}
        return new_args, new_kwargs

    def repack_module_input(self, value):
        if isinstance(value, list) or (isinstance(value, tuple) and len(value) == 1):
            value = [self.repack_value(v) for v in value]
        else:
            value = [self.repack_value(value)]
        return value

    def repack_model_output(self, value):
        if isinstance(value, (list, tuple)):
            value = self.repack_value(value)
        else:
            value = [self.repack_value(value)]
        return value

    def return_output(self, output):
        if isinstance(output, tuple(TYPES_TO_TRACE)):
            return replace(self, value_=output)
        else:
            return output

    def epilogue(self, inplace, output):
        if inplace:
            assert not isinstance(output, tuple)  # sanity check
            self.value_ = output
            return self
        else:
            return self.return_output(output)

    def _register_forward_pre_hooks(self, model: torch.nn.Module):

        def enter_namespace(module, input):
            module_name = None
            if not self.namespace_:  # root module
                module_name = ''
            else:
                supermodule = self.namespace_[-1][0]
                for name, mod in supermodule.named_modules():
                    # identify current module as a submodule
                    if mod is module:
                        # get module name as name of the identified submodule
                        module_name = name.split('.')
                        break
            input_list = self.repack_module_input(input)
            namespace = (module, module_name, input_list)
            self.namespace_.append(namespace)

        # Identify input to the whole model
        def model_input(module, input):
            if module is model:
                input_list = self.repack_module_input(input)
                self.trace_.model_input_list = input_list

        for mod in model.modules():
            if not isinstance(mod, torch.jit.ScriptModule):
                mod.register_forward_pre_hook(enter_namespace)
                mod.register_forward_pre_hook(model_input)

    def _register_forward_hooks(self, model):

        # input is possibly the Tracer
        def exit_namespace(module, input, output):
            assert self.namespace_[-1][0] is module  # check module is on the stack
            for trace_elem in self.trace_.trace_elem_list:
                if trace_elem.module_context_list[-1] is module:
                    if trace_elem.module_output is None:
                        output = self.repack_value(output)
                        trace_elem.module_output = output
            self.namespace_.pop()

        # Identity output of the whole model
        def model_output(module, input, output):
            if module is model:
                output = self.repack_model_output(output)
                self.trace_.model_output_list = output

        for mod in model.modules():
            if not isinstance(mod, torch.jit.ScriptModule):
                mod.register_forward_hook(exit_namespace)
                mod.register_forward_hook(model_output)

    def move_torch_args_to_kwargs(self, fn, fn_args, fn_kwargs):
        fn_stub = TORCH_FN_OVERRIDE_DICT[fn]
        fn_kwargs = getcallargs(fn_stub, *fn_args, **fn_kwargs)
        return [], fn_kwargs

    def update_inplace_output(self, out, fn_args, fn_kwargs):
        inplace = any([out is arg for arg in fn_args + list(fn_kwargs.values())])
        out = out.clone() if inplace and isinstance(out, Tensor) else out
        return out, inplace

    def __torch_function__(self, fn, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        args, kwargs = self.move_torch_args_to_kwargs(fn, args, kwargs)
        args, kwargs = self.repack_args_kwargs(args, kwargs)
        # get rid of out=None early on since it's often problematic
        if 'out' in kwargs and kwargs['out'] is None: del kwargs['out']
        out = fn(*args, **kwargs)
        out, inplace = self.update_inplace_output(out, args, kwargs)
        out = self.update_trace(fn, FnType.FUNCTION, args, kwargs, out)
        return self.epilogue(inplace, out)

    def update_trace(self, fn, fn_type, fn_args, fn_kwargs, fn_out):
        modules, m_names, m_inputs = zip(*self.namespace_)
        m_names = flatten(m_names)
        # remove first empty name
        m_names = m_names[1:]
        # take only innermost input list, corresponding to modules[-1]
        m_input_list = m_inputs[-1]
        # empty fn_args means it would be an empty tuple, which we don't want
        fn_args = [] if fn_args == () else fn_args
        # repack fn_out
        fn_out = self.repack_value(fn_out)
        # assign index to fn inputs and outputs
        fn_args_index = [self.trace_.index_from_val(a) for a in fn_args]
        fn_kwargs_index = {k: self.trace_.index_from_val(a) for k,a in fn_kwargs.items()}
        fn_out_index = self.trace_.index_from_val(fn_out, False)
        # generate trace element
        trace_elem = TraceElem(
            fn=fn,
            fn_type=fn_type,
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
            fn_out=fn_out,
            fn_args_index=fn_args_index,
            fn_kwargs_index=fn_kwargs_index,
            fn_out_index=fn_out_index,
            module_context_list=modules,
            prefix_list=m_names,
            module_input_list=m_input_list)
        # update trace
        self.trace_.trace_elem_list.append(trace_elem)
        # return fn_out with newly interned values
        return fn_out

    # https://stackoverflow.com/questions/3024925/create-a-with-block-on-several-context-managers
    def _trace_with_patches(self, model):
        with ExitStack() as stack:
            for mgr in PATCHES:
                stack.enter_context(mgr)
            model(self)

    def trace_model(self, model, wrap_torchscript=True):
        model = copy(model)
        if wrap_torchscript:
            torchscript_wrapper(model)
        self._register_forward_pre_hooks(model)
        self._register_forward_hooks(model)
        self._trace_with_patches(model)
        return self.trace_
