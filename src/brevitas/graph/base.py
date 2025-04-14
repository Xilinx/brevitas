# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
import inspect
from inspect import getcallargs
from typing import Any, Dict, Type

import torch
from torch.nn import Module
from torch.overrides import get_testing_overrides

# TODO: Deprecate PyTorch 1.11
try:
    from torch.nn.utils.parametrize import is_parametrized
    from torch.nn.utils.parametrize import register_parametrization
except ImportError:
    from brevitas.utils.torch_utils import is_parametrized
    register_parametrization = None

from brevitas.fx import GraphModule
from brevitas.fx import immutable_dict
from brevitas.fx import Node
from brevitas.graph.utils import *
from brevitas.utils.python_utils import islambda

__all__ = [
    'Transform',
    'PerInputTransform',
    'GraphTransform',
    'PerInputModuleToModuleByHook',
    'ModuleToModule',
    'InsertModuleCallAfter',
    'ModuleToModuleByName',
    'ModuleToModuleByClass',
    'ModuleInstanceToModuleInstance',
    'ModuleToModuleByInstance',
    'MethodToModule',
    'FnToModule',
    'CallableToModule']

_TORCH_TESTING_DICT = get_testing_overrides()


class Transform(ABC):

    @abstractmethod
    def apply(self, model: Module) -> Module:
        pass


class PerInputTransform(ABC):

    @abstractmethod
    def apply(self, model: Module, inp: torch.Tensor) -> Module:
        pass


class GraphTransform(Transform):

    @abstractmethod
    def apply(self, graph_model: GraphModule) -> GraphModule:
        pass


class UntilFixedPointGraphTransform(Transform):

    @abstractmethod
    def is_converged(self, graph_model: GraphModule) -> bool:
        pass

    def apply(self, graph_model: GraphModule) -> GraphModule:
        while not self.is_converged(graph_model):
            continue
        return graph_model


class PerInputModuleToModuleByHook(PerInputTransform, ABC):

    def __init__(self):
        self.input_size_map = {}
        self.hook_handlers = []

    @abstractmethod
    def register_hooks(self, model):
        pass

    @abstractmethod
    def replace_modules(self, model):
        pass

    def hook_fn(self, module, inp):
        if isinstance(inp, tuple):
            assert len(inp) == 1
            inp = inp[0]
        size = inp.size()
        if module in self.input_size_map.keys() and self.input_size_map[module] != size:
            raise RuntimeError("Layer called multiple times with different input sizes.")
        self.input_size_map[module] = size

    def cleanup(self):
        for hook in self.hook_handlers:
            hook.remove()
        self.hook_handlers = []
        self.input_size_map = {}

    def apply(self, model: Module, *model_args, **model_kwargs):
        self.register_hooks(model)
        model(*model_args, **model_kwargs)
        self.replace_modules(model)
        self.cleanup()
        return model


def _remove_parametrization_entries_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Keys for values related to parametrizations
    keys_to_remove = []
    # Keys/values corresponding to the original tensors before parametrizations
    keys_values_to_add = []
    # Iterate over state_dict identifying the keys related to parametrizations
    for key, value in state_dict.items():
        split_key = key.split(".")
        # Keys for values before parametrizations have the format "prefix.parametrizations.tensor_name.original"
        if len(split_key
              ) >= 3 and split_key[-3] == "parametrizations" and split_key[-1] == "original":
            tensor_name = split_key[-2]
            # Name of dictionary entry is "prefix.tensro_name"
            keys_values_to_add.append((".".join(split_key[:-3] + [tensor_name]), value))
        # Keys corresponding to the parametrizations attached to the model need to be removed
        # to make sure the dictionary can be loaded with no missing/unused keys
        # NOTE: For safety, an additional check could be added as this logic would not work if a model
        # without parametrizations has any key containing "parametrizations"
        if "parametrizations" in split_key:
            keys_to_remove.append(key)
    # Apply changes in-place to the state_dict
    for key in keys_to_remove:
        del state_dict[key]
    for key, value in keys_values_to_add:
        state_dict[key] = value
    return state_dict


class ModuleToModule(GraphTransform, ABC):

    def __init__(self, new_module_class, **kwargs):
        super().__init__()
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs

    def _map_origin_vars(self, vars: dict):
        return {k: v is not None if k == 'bias' else v for k, v in vars.items()}

    def _module_attributes(self, module):
        attrs = vars(module)
        # workaround since bias doesn't show up on vars of Linear
        if hasattr(module, 'bias'):
            attrs['bias'] = module.bias
        return attrs

    def _evaluate_new_kwargs(self, new_kwargs, old_module, name):
        update_dict = dict()
        for k, v in self.new_module_kwargs.items():
            if islambda(v):
                if name is not None:
                    # Two types of lambdas are admitted now, with/without the name of the module as input
                    if len(inspect.getfullargspec(v).args) == 2:
                        v = v(old_module, name)
                    elif len(inspect.getfullargspec(v).args) == 1:
                        v = v(old_module)
                else:
                    v = v(old_module)
            update_dict[k] = v
        new_kwargs.update(update_dict)
        return new_kwargs

    def _init_new_module(self, old_module: Module, name=None):
        # get attributes of original module
        new_kwargs = self._module_attributes(old_module)
        # transforms attribute of original module, e.g. bias Parameter -> bool
        new_kwargs = self._map_origin_vars(new_kwargs)
        # restrict to only values that are in the init of the new module
        new_module_signature_keys = signature_keys(self.new_module_class)
        new_kwargs = {k: v for k, v in new_kwargs.items() if k in new_module_signature_keys}
        # update with kwargs passed to the rewriter
        new_kwargs = self._evaluate_new_kwargs(new_kwargs, old_module, name)
        # init the new module
        new_module = self.new_module_class(**new_kwargs)
        return new_module

    def _replace_old_module(self, model, old_module, new_module, load_state_dict=True):
        replace_module(model, old_module, new_module)
        if load_state_dict:
            # work-around since weight_orig is from a previous quantization algorithm
            if hasattr(old_module, 'weight_orig'):
                new_module.register_buffer('weight_orig', old_module.weight_orig.data.cpu())
            if not is_parametrized(old_module):
                new_module.load_state_dict(old_module.state_dict())
            else:
                old_module_state_dict = old_module.state_dict()
                # If parametrizations are present in old_module, the state_dict needs
                # to be processed beforehand
                old_module_state_dict = _remove_parametrization_entries_state_dict(
                    old_module_state_dict)
                # Strict can be set to True, since potential parametrizations were
                # accounted for
                new_module.load_state_dict(old_module_state_dict)
                # If the old module is parametrized, these need to be transferred to the new module.
                # The method transfer_parametrizations_and_params as it can result in parameter ties
                # being broken
                # NOTE: unsafe is set to True for efficiency, as the checks should have been done
                # when first registering the parametrization to old_module
                for tensor_name in old_module.parametrizations:
                    for param_func in old_module.parametrizations[tensor_name]:
                        register_parametrization(new_module, tensor_name, param_func, unsafe=True)


class InsertModuleCallAfter(GraphTransform):

    def __init__(self, module_name, node, node_to_exclude=()):
        self.module_name = module_name
        self.node = node
        self.node_to_exclude = node_to_exclude

    def apply(self, graph_model: GraphModule) -> GraphModule:
        with graph_model.graph.inserting_after(self.node):
            quant_identity_node = graph_model.graph.call_module(self.module_name, args=(self.node,))
        replace_all_uses_except(
            self.node, quant_identity_node, [quant_identity_node] + list(self.node_to_exclude))
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class ModuleInstanceRegisterParametrization(Transform):
    r"""Transform to register a parametrization to a given parameter of a
    module.

    Args:
        module (nn.Module): module on which to register the
            parametrization
        tensor_name: (str): name of the :class:`torch.nn.Parameter` of
            module which is to be parametrized
        transform_module (nn.Module): the parametrization to
            register
    """

    def __init__(self, module: Module, tensor_name: str, transform_module: Module) -> None:
        self.module = module
        self.tensor_name = tensor_name
        self.transform_module = transform_module

    # TODO: Unify inferfaces with ModuleInstanceToModuleInstance for
    # compatibility with fix_rewriter
    @property
    def old_module_instance(self):
        return self.module

    @old_module_instance.setter
    def old_module_instance(self, old_module_instance):
        self.module = old_module_instance

    def apply(self, model: GraphModule) -> GraphModule:
        for module in model.modules():
            if module is self.module:
                # register the parametrization to module
                register_parametrization(module, self.tensor_name, self.transform_module)
                break
        return model


class ModuleInstanceTransformTensor(Transform):
    r"""Transform to transform in-place a given parameter of a module

    Args:
        module (nn.Module): parent module of the parameter to be transformed
        tensor_name (str): name of the :class:`torch.nn.Parameter` of
            module which is to be transformed
        transform_module (nn.Module): module defining the transformation to apply
            to the tensor
    """

    def __init__(
        self,
        module: Module,
        tensor_name: str,
        transform_module: Module,
    ):
        self.module = module
        self.tensor_name = tensor_name
        self.transform_module = transform_module

    # TODO: Unify inferfaces with ModuleInstanceToModuleInstance for
    # compatibility with fix_rewriter
    @property
    def old_module_instance(self):
        return self.module

    @old_module_instance.setter
    def old_module_instance(self, old_module_instance):
        self.module = old_module_instance

    def apply(self, model: GraphModule) -> GraphModule:
        for module in model.modules():
            if module is self.module:
                # This check is needed to apply the change in the parameters
                # when the model is offloaded
                # TODO: Move outside the apply function
                if hasattr(module, 'allocate_params'):
                    module.allocate_params(module)
                tensor = getattr(module, self.tensor_name).data
                tensor = self.transform_module(tensor)
                # Modify the weights in-place
                setattr(module, self.tensor_name, torch.nn.Parameter(tensor))

                if hasattr(module, 'offload_params'):
                    module.offload_params(module)
                break
        return model


class ModuleInstanceWrapModule(Transform):
    r"""Transform to replace a module by a wrapper module which has the original
        one as a submodule

    Args:
        old_module_instance (nn.Module): module to be wrapped
        wrapper_class (type): class of the wrapper for old_module_instance
        module_attribute (str): name of the parameter to pass the original
            module to the constructor of wrapper_class
        kwargs_wrapper (dict, optional): dictionary with the constructor
            arguments for wrapper_class
    """

    def __init__(
            self,
            old_module_instance: Module,
            wrapper_class: Type[Module],
            module_attribute: str,
            kwargs_wrapper: Dict[str, Any]):
        self.old_module_instance = old_module_instance
        self.wrapper_class = wrapper_class
        self.module_attribute = module_attribute
        self.kwargs_wrapper = kwargs_wrapper

    def apply(self, model: GraphModule) -> GraphModule:
        for old_module in model.modules():
            if old_module is self.old_module_instance:
                kwargs = {self.module_attribute: self.old_module_instance}
                kwargs.update(self.kwargs_wrapper)
                new_module_instance = self.wrapper_class(**kwargs)
                # init the new module based on the old one
                replace_module(model, old_module, new_module_instance)
                break
        return model


class ModuleInstanceToModuleInstance(Transform):

    def __init__(self, old_module_instance, new_module_instance):
        self.old_module_instance = old_module_instance
        self.new_module_instance = new_module_instance

    def apply(self, model: GraphModule) -> GraphModule:
        for old_module in model.modules():
            if old_module is self.old_module_instance:
                # init the new module based on the old one
                replace_module(model, old_module, self.new_module_instance)
                break
        return model


class ModuleToModuleByName(ModuleToModule):

    def __init__(self, old_module_name, new_module_class, **kwargs):
        super().__init__(new_module_class, **kwargs)
        self.old_module_name = old_module_name

    def apply(self, model: GraphModule) -> GraphModule:
        for name, old_module in model.named_modules():
            if name == self.old_module_name:
                # init the new module based on the old one
                new_module = self._init_new_module(old_module)
                self._replace_old_module(model, old_module, new_module)
                break
        return model


class ModuleToModuleByInstance(ModuleToModule):

    def __init__(self, old_module_instance, new_module_class, **kwargs):
        super().__init__(new_module_class, **kwargs)
        self.old_module_instance = old_module_instance

    def apply(self, model: GraphModule) -> GraphModule:
        for name, old_module in model.named_modules():
            if old_module is self.old_module_instance:
                # init the new module based on the old one
                new_module = self._init_new_module(old_module, name)
                self._replace_old_module(model, old_module, new_module)
                break
        return model


class ModuleToModuleByClass(ModuleToModule):

    def __init__(self, old_module_class, new_module_class, **kwargs):
        super().__init__(new_module_class, **kwargs)
        self.old_module_class = old_module_class

    def apply(self, model: GraphModule) -> GraphModule:
        old_new_module_dict = {}
        for old_module in model.modules():
            # check for equality, not inheritance
            if type(old_module) == self.old_module_class:
                # init the new module based on the old one
                new_module = self._init_new_module(old_module)
                # register modules pair to be replaced
                old_new_module_dict[old_module] = new_module
        # replace all pairs registered
        for old_module, new_module in old_new_module_dict.items():
            self._replace_old_module(model, old_module, new_module)
        return model


class CallableToModule(GraphTransform, ABC):

    def __init__(self, old_callable, new_module_class, **kwargs):
        super().__init__()
        self.old_callable = old_callable
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs

    @abstractmethod
    def match_node(self, node: Node):
        pass

    def split_kwargs(self, node: Node):
        new_module_keys = signature_keys(self.new_module_class)
        node_kwargs = dict(node.kwargs)
        node_kwargs_keys = list(node_kwargs.keys())
        module_kwargs = {k: node_kwargs.pop(k) for k in node_kwargs_keys if k in new_module_keys}
        return node_kwargs, module_kwargs

    def move_node_args_to_kwargs(self, node: Node):
        """Move non Node args to kwargs, as long as we can resolve the fn signature somehow"""
        fn = node.target
        if fn in _TORCH_TESTING_DICT:
            fn = _TORCH_TESTING_DICT[fn]
        try:
            fn_kwargs = getcallargs(fn, *node.args, **node.kwargs)
            fn_args = []
            for k, a in list(fn_kwargs.items()):
                if isinstance(a, Node):
                    fn_args.append(fn_kwargs.pop(k))
                else:
                    break
            node.args = tuple(fn_args)
            node.kwargs = immutable_dict(fn_kwargs)
        except TypeError:
            pass

    def rewrite_node(self, node: Node, graph_model: GraphModule):
        module_name = node.name
        assert module_name not in dict(graph_model.named_modules()).keys()
        self.move_node_args_to_kwargs(node)
        node_kwargs, module_kwargs = self.split_kwargs(node)
        module = self.new_module_class(**module_kwargs, **self.new_module_kwargs)
        node.target = module_name
        node.op = 'call_module'
        node.kwargs = immutable_dict(node_kwargs)
        set_module(graph_model, module, module_name)

    def apply(self, graph_model: GraphModule) -> GraphModule:
        for node in graph_model.graph.nodes:
            if self.match_node(node):
                self.rewrite_node(node, graph_model)
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class FnToModule(CallableToModule):

    def match_node(self, node: Node) -> bool:
        return node.op == 'call_function' and node.target is self.old_callable

    def move_node_args_to_kwargs(self, node: Node):
        super().move_node_args_to_kwargs(node)
        # Moving to stateful modules, we remove the 'training' argument if it is passed to the
        # functional version of the layer since it is not needed anymore
        kwargs = dict(node.kwargs)
        if 'training' in kwargs:
            del kwargs['training']
        node.kwargs = immutable_dict(kwargs)


class MethodToModule(CallableToModule):

    def match_node(self, node: Node) -> bool:
        return node.op == 'call_method' and node.target is self.old_callable

    def move_node_args_to_kwargs(self, node: Node):
        if 'self' in node.kwargs:
            node_kwargs = dict(node.kwargs)
            slf = node_kwargs.pop('self')
            node.kwargs = immutable_dict(node_kwargs)
            node.args = tuple([slf] + list(node.args))
