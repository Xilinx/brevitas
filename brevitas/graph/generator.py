from typing import List
from copy import copy
from inspect import isclass

from torch import Tensor
from torch.nn import Parameter
from torch.nn import Module, Sequential, ModuleList, ModuleDict

from .module import CodegenModule, Index, Instruction
from .tracer.trace import Trace, TraceElem, FnType
from .utils import module_class_name, flatten


class ModuleGenerator(object):

    default_gen_module_allowlist = ['torch.nn', 'brevitas.nn']
    default_gen_module_blocklist = [Sequential, ModuleList, ModuleDict]

    def __init__(
            self,
            gen_module_allowlist=tuple(default_gen_module_allowlist),
            gen_module_blocklist=tuple(default_gen_module_blocklist)):
        self.gen_module_allowlist = list(gen_module_allowlist)
        self.gen_module_blocklist = list(gen_module_blocklist)
        self._generated_allowlist_modules = []

    def _parent_in_gen_allowlist(self, trace_elem):
        # find parent that has to be preserved as-is, if any
        output = None
        # context is all parent modules, from the inner most to the outer most
        context = reversed(trace_elem.module_context_list[:-1])
        for c in context:
            for gma in self.gen_module_allowlist:
                if isinstance(gma, str) and module_class_name(c).startswith(gma):
                    if not self._is_module_in_gen_blocklist(c):
                        output = c
                elif isclass(gma) and isinstance(c, gma):
                    if not self._is_module_in_gen_blocklist(c):
                        output = c
        return output

    def _is_module_in_gen_list(self, module, gen_list):
        output = False
        for gm in gen_list:
            output |= isinstance(gm, str) and module_class_name(module).startswith(gm)
            output |= isclass(gm) and isinstance(module, gm)
        return output

    def _is_module_in_gen_blocklist(self, module):
        return self._is_module_in_gen_list(module, self.gen_module_blocklist)

    def _is_module_in_gen_allowlist(self, module):
        return self._is_module_in_gen_list(module, self.gen_module_allowlist)

    def _is_module_already_gen(self, module, trace_elem):
        is_already_gen = False
        # check that the module itself hasn't been generated yet
        # input and output have to be the same
        # otherwise it's a different invokation of the same module
        for m, i_list, o in self._generated_allowlist_modules:
            same_module = m is module
            same_input = all([i is mi for i, mi in zip(i_list, trace_elem.module_input_list)])
            same_output = o is trace_elem.module_output
            is_already_gen |= same_module and same_input and same_output
        # check that none of the parents has been already preserved as-is
        for c in trace_elem.module_context_list:  # TODO same input and output across parents
            for m in self._generated_allowlist_modules:
                is_already_gen |= c is m
        return is_already_gen

    def _add_module(self, module: Module, prefix_list, model: Module):
        supermodule = model
        module_name = prefix_list[-1]
        prefix_list = prefix_list[:-1]  # exclude module name
        for prefix in prefix_list:
            submodule_names, submodules = zip(*list(supermodule.named_modules()))
            if prefix in submodule_names:
                supermodule = supermodule._modules[prefix]
            else:
                submodule = Module()
                supermodule.add_module(prefix, submodule)
                supermodule = submodule
        supermodule.add_module(module_name, module)

    def _module_instruction(
            self, module: Module, trace_elem: TraceElem, trace: Trace, output_model: Module):
        self._add_module(module, trace_elem.prefix_list, output_model)
        iil = [trace.index_from_val(i) for i in trace_elem.module_input_list]
        module_output_index = trace.index_from_val(trace_elem.module_output)
        inst = Instruction(module_output_index, module, FnType.MODULE, iil, {}, trace_elem.prefix)
        return inst

    def _torch_fn_instruction(self, trace_elem: TraceElem):
        fn_args = trace_elem.fn_args_index
        fn_kwargs = trace_elem.fn_kwargs_index
        fn = trace_elem.fn
        fn_type = trace_elem.fn_type
        output_index = trace_elem.fn_out_index
        return Instruction(output_index, fn, fn_type, fn_args, fn_kwargs, trace_elem.prefix)

    def _tensor_fn_instruction(self, trace_elem: TraceElem, trace: Trace):
        fn = trace_elem.fn
        fn_type = trace_elem.fn_type
        fn_args = trace_elem.fn_args_index
        fn_kwargs = trace_elem.fn_kwargs_index
        output_index = trace_elem.fn_out_index
        return Instruction(output_index, fn, fn_type, fn_args, fn_kwargs, trace_elem.prefix)

    def _gen_schedule(self, trace: Trace, gen_model: Module):
        schedule = []
        for trace_elem in trace.trace_elem_list:
            module = trace_elem.module_context_list[-1]
            # if a parent is supposed to be preserved as is
            parent_in_allowlist = self._parent_in_gen_allowlist(trace_elem)
            # If the module wrapping this fn is supposed to be preserved as-is
            in_allowlist = self._is_module_in_gen_allowlist(module)
            in_allowlist &= not self._is_module_in_gen_blocklist(module)
            # if either the wrapping module or parents have been already preserved as-is
            m_already_gen = self._is_module_already_gen(module, trace_elem)
            inst = None
            if parent_in_allowlist is not None and not m_already_gen:
                ctx = (module, trace_elem.module_input_list, trace_elem.module_output)
                self._generated_allowlist_modules.append(ctx)
                inst = self._module_instruction(parent_in_allowlist, trace_elem, trace, gen_model)
            elif in_allowlist and not m_already_gen:
                ctx = (module, trace_elem.module_input_list, trace_elem.module_output)
                self._generated_allowlist_modules.append(ctx)
                inst = self._module_instruction(module, trace_elem, trace, gen_model)
            elif in_allowlist and m_already_gen:
                continue
            else:
                if trace_elem.fn_type == FnType.FUNCTION:
                    inst = self._torch_fn_instruction(trace_elem)
                elif trace_elem.fn_type == FnType.METHOD:
                    inst = self._tensor_fn_instruction(trace_elem, trace)
                elif trace_elem.fn_type == FnType.ATTRIBUTE:
                    inst = self._tensor_fn_instruction(trace_elem, trace)
                elif trace_elem.fn_type == FnType.SCRIPTMODULE:
                    inst = self._module_instruction(trace_elem.fn, trace_elem, trace, gen_model)
                assert inst is not None
            schedule.append(inst)
        return schedule

    # constants are all inputs not computed as outputs, excluding the model input
    def _gen_constants_parameters(self, schedule: List[Instruction], trace: Trace):
        output_index_list = [inst.output_index for inst in schedule]
        input_index_list = flatten([i for inst in schedule for i in inst.input_args_index_list])
        input_index_list += flatten([i for inst in schedule for n, i in inst.input_kwargs_index_dict.items()])
        # remove topmost inputs from constants
        const_index_list = [i for i in input_index_list if i not in trace.model_input_index_list]
        # remove values generated as output from constants
        const_index_list = [i for i in const_index_list if i not in output_index_list]
        # filter out parameters
        consts = {c: trace.index_map[c] for c in const_index_list if not isinstance(c, Parameter)}
        params = {c: trace.index_map[c] for c in const_index_list if isinstance(c, Parameter)}
        # only scalar tensors are accepted as constants, otherwise throw an error
        consts = {k: v.item() if isinstance(v, Tensor) else v for k, v in consts.items()}
        if None in consts.keys():
            raise RuntimeError("Something went wrong")
        return consts, params

    def gen_model(self, trace: Trace):
        output_model = CodegenModule()
        schedule = self._gen_schedule(trace, output_model)
        constants, parameters = self._gen_constants_parameters(schedule, trace)
        output_model.schedule = schedule
        output_model.parameters_index_dict = parameters
        output_model.constants_index_dict = constants
        output_model.input_index_list = trace.model_input_index_list
        output_model.output_index_list = trace.model_output_index_list
        return output_model