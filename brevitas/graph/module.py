from typing import List, Union, Callable, Dict, Any
from dataclasses import dataclass
from brevitas.utils.python_utils import AutoName
from enum import auto

from torch.nn import Module, Parameter

from .utils import flatten


class FnType(AutoName):
    FUNCTION = auto()
    ATTRIBUTE = auto()
    METHOD = auto()
    SCRIPTMODULE = auto()
    MODULE = auto()


def _set_module(model, module, prefix):
    supermodule = model
    prefix_list = prefix.split('.')
    module_name = prefix_list[-1]
    prefix_list = prefix_list[:-1]  # exclude module name
    for prefix in prefix_list:
        if prefix:  # exclude empty prefix
            supermodule = supermodule._modules[prefix]
    supermodule._modules[module_name] = module


def _replace_module(model, old_module, new_module):
    name_list = []  # list of all references to old_module
    for name, module in model.named_modules():
        if old_module is module:
            name_list.append(name)
    for name in name_list:
        _set_module(model, new_module, name)


@dataclass(eq=True, frozen=True)
class Index:
    id: int

    def __str__(self):
        return str(f"_{self.id}")


@dataclass
class Instruction(object):
    output_index: Index
    fn: Union[str, Callable]
    fn_type: FnType
    input_args_list: List[Any]
    input_kwargs_dict: Dict[str, Any]
    prefix: str

    @property
    def fn_name(self):
        if isinstance(self.fn, str):
            return self.fn
        elif isinstance(self.fn, Module):
            return self.fn.__class__.__name__
        else:
            return self.fn.__name__

    def __str__(self):
        return (f"{[self.output_index]} "
                f"{self.fn_name} "
                f"{[i for i in self.input_args_list]} "
                f"{[i for i in self.input_kwargs_dict.items()]} "
                f"{self.prefix}")


class CodegenModule(Module):

    def __init__(self):
        super().__init__()
        self.schedule: List[Instruction] = []
        self.input_index_list = List[Index]
        self.output_index_list = List[Index]

    def fresh_index(self):
        max_id = 0
        for inst in self.schedule:
            max_id = max(
                max_id,
                inst.output_index,
                *[ii.id for ii in inst.input_args_list],
                *[ii.id for ii in inst.input_kwargs_dict.values()])
        return Index(max_id)

    def inst_successor_list(self, inst):
        successors = []
        for i in self.schedule:
            if inst.output_index in i.input_args_list or \
                    inst.output_index in i.input_kwargs_dict.values():
                successors.append(i)
        return successors

    def module_use_dict(self):
        use_dict = {}
        for inst in self.schedule:
            if isinstance(inst.fn, Module):
                if inst.fn in use_dict.keys():
                    use_dict[inst.fn].append(inst)
                else:
                    use_dict[inst.fn] = [inst]
        return use_dict

    def gc_state(self, state, current_schedule):
        # set of values needed by future instructions
        index_set_needed = set(flatten(self.output_index_list))
        for inst in current_schedule:
            args_index_list = flatten(inst.input_args_list)
            kwargs_index_list = flatten(list(inst.input_kwargs_dict.values()))
            args_index_list = [i for i in args_index_list if isinstance(i, Index)]
            kwargs_index_list = [i for i in kwargs_index_list if isinstance(i, Index)]
            index_set_needed.update(args_index_list)
            index_set_needed.update(kwargs_index_list)
        for index in list(state.keys()):
            if index not in index_set_needed:
                del state[index]

    def input_args_from_state(self, state, v):
        if isinstance(v, list):
            return [self.input_args_from_state(state, vv) for vv in v]
        elif isinstance(v, tuple):
            return tuple(self.input_args_from_state(state, vv) for vv in v)
        elif isinstance(v, Index):
            return state[v]
        else:
            return v

    def input_kwargs_from_state(self, state, kwargs_dict):
        return {n: self.input_args_from_state(state, v) for n, v in kwargs_dict.items()}

    def update_state_from_output(self, state, index, value):
        if isinstance(index, (list, tuple)):
            for i, v in zip(index, value):
                self.update_state_from_output(state, i, v)
        else:
            state[index] = value

    def compute_inst(self, inst: Instruction, state):
        # get needed args and kwargs from state
        args = self.input_args_from_state(state, inst.input_args_list)
        kwargs = self.input_kwargs_from_state(state, inst.input_kwargs_dict)
        if inst.fn_type == FnType.METHOD:
            obj = kwargs.pop('self')
            fn = getattr(obj, inst.fn)
            output_value = fn(*args, **kwargs)
        elif inst.fn_type == FnType.ATTRIBUTE:
            obj = kwargs.pop('self')
            output_value = getattr(obj, inst.fn)
        else:
            output_value = inst.fn(*args, **kwargs)
        return output_value

    def model_output_from_state(self, state):
        output = {idx: val for idx, val in state.items() if idx in self.output_index_list}
        sorted_output = [output[idx] for idx in self.output_index_list]
        if len(sorted_output) == 1:
            return sorted_output[0]
        else:
            return tuple(sorted_output)

    def forward(self, *args):  # TODO deal with input kwargs
        state: Dict[Index, Any] = {}
        assert len(args) == len(self.input_index_list), "Unexpected number of inputs"
        input_kwargs = {index: val for index, val in zip(self.input_index_list, args)}
        state.update(input_kwargs)  # init with input kwargs
        for i, inst in enumerate(self.schedule):
            output_value = self.compute_inst(inst, state)
            state[inst.output_index] = output_value
            # remove from state references to values not needed anymore
            self.gc_state(state, self.schedule[i + 1:])
        return self.model_output_from_state(state)
