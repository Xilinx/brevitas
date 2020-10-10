from inspect import signature
from abc import abstractmethod, ABC
from copy import deepcopy

import torch
from torch.nn import Module
from torch.nn import Conv2d, Conv1d, Linear

from brevitas import nn as qnn
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.utils import merge_bn
from .module import CodegenModule, _replace_module, Instruction, FnType, _set_module


def _signature_keys(module_class):
    return signature(module_class).parameters.keys()


class Rewriter(ABC):
    _module_prefix_id_map = {}

    @abstractmethod
    def apply(self, model: CodegenModule) -> CodegenModule:
        pass

    def module_id(self, name):
        if name in self._module_prefix_id_map:
            self._module_prefix_id_map[name] += 1
        else:
            self._module_prefix_id_map[name] = 0
        return self._module_prefix_id_map[name]


class InstRewriter(Rewriter, ABC):

    @abstractmethod
    def rewrite_inst(self, inst: Instruction, model: CodegenModule):
        pass

    def apply(self, model: CodegenModule):
        for inst in model.schedule:
            self.rewrite_inst(inst, model)
        return model


class ModuleToModuleRewriter(Rewriter, ABC):

    def __init__(
            self,
            old_module_class,
            new_module_class,
            old_module_cond=lambda old_module: True, **kwargs):
        super().__init__()
        self.old_module_class = old_module_class
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs
        self.old_module_cond = old_module_cond

    def map_origin_vars(self, vars: dict):
        return {k: v is not None if k == 'bias' else v for k, v in vars.items()}

    def module_attributes(self, module):
        attrs = vars(module)
        # workaround since bias doesn't show up on vars of Linear
        if hasattr(module, 'bias'):
            attrs['bias'] = module.bias
        return attrs

    def _init_new_module(self, old_module: Module):
        # get attributes of original module
        new_kwargs = self.module_attributes(old_module)
        # transforms attribute of original module, e.g. bias Parameter -> bool
        new_kwargs = self.map_origin_vars(new_kwargs)
        # restrict to only values that are in the init of the new module
        new_module_signature_keys = _signature_keys(self.new_module_class)
        new_kwargs = {k: v for k, v in new_kwargs.items() if k in new_module_signature_keys}
        # update with kwargs passed to the rewriter
        new_kwargs.update(self.new_module_kwargs)
        # init the new module
        new_module = self.new_module_class(**new_kwargs)
        return new_module

    def _rewrite_schedule(self, old_module_use_list, new_module):
        # replace old module references in the compute graph
        for inst in old_module_use_list:
            inst.fn = new_module

    def _rewrite_model(self, model, old_new_module_dict):
        for old_module, new_module in old_new_module_dict.items():
            _replace_module(model, old_module, new_module)
            new_module.load_state_dict(old_module.state_dict())

    def apply(self, model: CodegenModule):
        old_new_module_dict = {}
        for old_module in model.modules():
            # check for equality, not inheritance
            if type(old_module) == self.old_module_class and self.old_module_cond(old_module):
                # init the new module based on the old one
                new_module = self._init_new_module(old_module)
                # rewrite compute schedule
                old_module_use_list = model.module_use_dict()[old_module]
                self._rewrite_schedule(old_module_use_list, new_module)
                # register modules pair to be replaced
                old_new_module_dict[old_module] = new_module
        # replace all pairs registered
        self._rewrite_model(model, old_new_module_dict)
        return model


class MergeBatchNorm2d(Rewriter):

    merge_layers = [QuantWBIOL, Conv2d, Conv1d, Linear]

    def apply(self, model: CodegenModule):
        for inst in model.schedule:
            if type(inst.fn) == torch.nn.BatchNorm2d:
                num_args = len(inst.input_args_list)
                num_kwargs = len(inst.input_kwargs_dict.items())
                assert num_args + num_kwargs == 1
                try:
                    bn_input_index = inst.input_args_list[0]
                except:
                    bn_input_index = list(inst.input_kwargs_dict.values())[0]
                inst.merge_into_predecessor = None
                inst.fn.merged_inst = []
                # check that the predecessor doesn't feed into other layers
                can_merge = True
                for predecessor in model.schedule:
                    if predecessor.fn is not inst.fn:
                        for pii in predecessor.input_args_list:
                            for iii in inst.input_args_list:
                                can_merge &= pii != iii
                # identify predecessor and merge bn into it
                for predecessor in model.schedule:
                    if predecessor.output_index == bn_input_index:
                        if any([isinstance(predecessor.fn, l) for l in self.merge_layers]):
                            merge_bn(predecessor.fn, inst.fn)
                            inst.merge_into_predecessor = predecessor
                            inst.fn.merged_inst.append(True)
                        else:
                            inst.fn.merged_inst.append(False)
                # update output index of the predecessor and mark inst as to remove
                if inst.merge_into_predecessor is not None:
                    predecessor = inst.merge_into_predecessor
                    predecessor.output_index = inst.output_index
                    inst.output_index = None
                else:
                    del inst.merge_into_predecessor
        # update schedule to remove merged inst
        model.schedule = [i for i in model.schedule if i.output_index is not None]
        # update module hierarchy to remove merged bn
        for m in list(model.modules()):
            cond = lambda x: not hasattr(x, 'merged_inst') or not all(x.merged_inst)
            m._modules = {name: sm for name, sm in m._modules.items() if cond(sm)}
        # remove leftover metadata on bn that hasn't been merged
        for m in model.modules():
            if hasattr(m, 'merged_inst'):
                del m.merged_inst
        return model


class DuplicateSharedStatelessModule(Rewriter):

    def apply(self, model: CodegenModule):
        for m, use_list in model.module_use_dict().items():
            is_stateful = list(m.parameters()) or list(m.buffers())
            if len(use_list) > 1 and not is_stateful:
                for use_inst in use_list[1:]:
                    dup_m = deepcopy(m)
                    use_inst.fn = dup_m
                    model.add_module(str(use_inst.output_index), dup_m)
        return model


class DisableBreakingReturnQuantTensor(Rewriter):

    def apply(self, model: CodegenModule):
        for inst in model.schedule:
            if hasattr(inst.fn, 'return_quant_tensor') and inst.fn.return_quant_tensor:
                s_list = model.inst_successor_list(inst)
                cond = lambda s: (((s.fn_type == FnType.MODULE or s.fn_type == FnType.SCRIPTMODULE)
                                   and hasattr(s.fn, 'accept_quant_tensor')
                                   or (s.fn_type == FnType.ATTRIBUTE or s.fn_type == FnType.METHOD)
                                   and hasattr(QuantTensor, s.fn))
                                  and not s.fn_type == FnType.FUNCTION)
                if not all([cond(s) for s in s_list]):
                    inst.fn.return_quant_tensor = False
        return model


class CallableToModuleRewriter(Rewriter, ABC):

    def __init__(
            self,
            old_callable,
            new_module_class,
            inst_cond=lambda old_inst: True,
            inst_postprocess=lambda new_inst: new_inst,
            **kwargs):
        super().__init__()
        self.old_callable = old_callable
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs
        self.inst_cond = inst_cond
        self.inst_postprocess = inst_postprocess

    @abstractmethod
    def match_inst(self, inst):
        pass

    def new_module_prefix(self, inst):
        module_prefix = inst.prefix + '.' + inst.fn_name
        module_prefix = module_prefix + str(self.module_id(module_prefix))
        return module_prefix

    def update_kwargs(self, inst: Instruction):
        new_module_keys = _signature_keys(self.new_module_class)
        old_fn_kwargs = inst.input_kwargs_dict
        old_fn_kwargs_keys = list(old_fn_kwargs.keys())
        extracted = {k: old_fn_kwargs.pop(k) for k in old_fn_kwargs_keys if k in new_module_keys}
        self.new_module_kwargs.update(extracted)

    def gen_new_module(self, inst, model):
        module_prefix = self.new_module_prefix(inst)
        self.update_kwargs(inst)
        module = self.new_module_class(**self.new_module_kwargs)
        _set_module(model, module, module_prefix)
        return module

    def rewrite_inst(self, inst, model):
        module = self.gen_new_module(inst, model)
        inst.fn = module
        inst.fn_type = FnType.MODULE
        self.inst_postprocess(inst)

    def apply(self, model: CodegenModule) -> CodegenModule:
        for inst in model.schedule:
            if self.match_inst(inst):
                self.rewrite_inst(inst, model)
        return model


class TorchFnToModuleRewriter(CallableToModuleRewriter):

    def match_inst(self, inst) -> bool:
        return (self.inst_cond(inst)
                and inst.fn_type == FnType.FUNCTION
                and inst.fn == self.old_callable)


class TensorMethodToModuleRewriter(CallableToModuleRewriter):

    def match_inst(self, inst) -> bool:
        return (self.inst_cond(inst)
                and inst.fn_type == FnType.METHOD
                and inst.fn == self.old_callable)

    def update_kwargs(self, inst: Instruction):
        super(TensorMethodToModuleRewriter, self).update_kwargs(inst)
        tensor = inst.input_kwargs_dict.pop('self')
        inst.input_args_list = [tensor] + inst.input_args_list


class MeanToAdaptiveAvgPool2d(TensorMethodToModuleRewriter):

    def __init__(self):
        super(MeanToAdaptiveAvgPool2d, self).__init__(
            old_callable='mean',
            new_module_class=qnn.QuantAdaptiveAvgPool2d,
            inst_cond=MeanToAdaptiveAvgPool2d.match_mean_cond,
            inst_postprocess=MeanToAdaptiveAvgPool2d.postprocess_inst,
            output_size=(1, 1))

    @staticmethod
    def match_mean_cond(inst):
        return ([2, 3] in inst.input_args_list
                or (2, 3) in inst.input_args_list
                or 'dim' in inst.input_kwargs_dict
                and (inst.input_kwargs_dict['dim'] == (2, 3)
                     or inst.input_kwargs_dict['dim'] == [2, 3]))

    @staticmethod
    def postprocess_inst(inst):
        if 'dim' in inst.input_kwargs_dict:
            del inst.input_kwargs_dict['dim']
        elif (2, 3) in inst.input_args_list:
            inst.input_args_list.remove((2, 3))
        elif [2, 3] in inst.input_args_list:
            inst.input_args_list.remove([2, 3])
        # TODO this is a hack, a separate view instruction should be generated
        original_forward = inst.fn.forward
        def forward_with_view(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            return out.view(int(out.size(0)), -1)
        inst.fn.forward = forward_with_view


class InplaceMeanToAdaptiveAvgPool2d(TensorMethodToModuleRewriter):

    def __init__(self):
        super(InplaceMeanToAdaptiveAvgPool2d, self).__init__(
            old_callable='mean_',
            new_module_class=qnn.QuantAdaptiveAvgPool2d,
            inst_cond=MeanToAdaptiveAvgPool2d.match_mean_cond,
            inst_postprocess=MeanToAdaptiveAvgPool2d.postprocess_inst,
            output_size=(1, 1))


class RewriterList(Rewriter):

    def __init__(self, rewriter_list):
        self.rewriter_list = rewriter_list

    def apply(self, model: CodegenModule):
        for rewriter in self.rewriter_list:
            rewriter.apply(model)
        return model
