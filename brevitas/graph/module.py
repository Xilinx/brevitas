from typing import List, Union, Callable, Dict, Any
from dataclasses import dataclass
from brevitas.utils.python_utils import AutoName
from enum import auto

from torch.nn import Module, Parameter

from brevitas.quant_tensor import QuantTensor
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
    input_index_list: List[Index]
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
                f"{[i for i in self.input_index_list]} "
                f"{self.prefix}")


class CodegenModule(Module):

    def __init__(self):
        super().__init__()
        self.schedule: List[Instruction] = []
        self.constants_index_dict: Dict[Index, Any] = {}
        self.parameters_index_dict: Dict[Index, Parameter] = {}
        self.input_index_list = List[Index]
        self.output_index_list = List[Index]

    def fresh_index(self):
        max_id = 0
        for inst in self.schedule:
            max_id = max(
                max_id,
                inst.output_index,
                *[ii.id for ii in inst.input_index_list])
        return Index(max_id)

    def inst_successor_list(self, inst):
        successors = []
        for i in self.schedule:
            if inst.output_index in i.input_index_list:
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
            index_set_needed.update(flatten(inst.input_index_list))
        for index in list(state.keys()):
            if index not in index_set_needed:
                del state[index]

    def input_from_state(self, state, index):
        if isinstance(index, list):
            return [self.input_from_state(state, i) for i in index]
        elif isinstance(index, tuple):
            return tuple(self.input_from_state(state, i) for i in index)
        else:
            return state[index]

    def update_state_from_output(self, state, index, value):
        if isinstance(index, (list, tuple)):
            for i, v in zip(index, value):
                self.update_state_from_output(state, i, v)
        else:
            state[index] = value

    def compute_inst(self, inst: Instruction, state):
        # get needed args from state, assuming correct ordering has been preserved
        args = self.input_from_state(state, inst.input_index_list)
        if inst.fn_type == FnType.METHOD:
            # convention is last arg is the tensor the fn is called on
            fn = getattr(args.pop(), inst.fn)
            output_value = fn(*args)
        elif inst.fn_type == FnType.ATTRIBUTE:
            output_value = getattr(args.pop(), inst.fn)
        else:
            output_value = inst.fn(*args)
        return output_value

    def model_output_from_state(self, state):
        output = {idx: val for idx, val in state.items() if idx in self.output_index_list}
        sorted_output = [output[idx] for idx in self.output_index_list]
        if len(sorted_output) == 1:
            return sorted_output[0]
        else:
            return tuple(sorted_output)

    def forward(self, *args):  # TODO deal with kwargs
        state: Dict[Index, Any] = {}
        assert len(args) == len(self.input_index_list), "Unexpected number of inputs"
        input_kwargs = {index: val for index, val in zip(self.input_index_list, args)}
        state.update(input_kwargs)  # init with input kwargs
        state.update(self.constants_index_dict)  # init with model constants
        for i, inst in enumerate(self.schedule):
            output_value = self.compute_inst(inst, state)
            state[inst.output_index] = output_value
            # remove from state references to values not needed anymore
            self.gc_state(state, self.schedule[i + 1:])
        return self.model_output_from_state(state)
