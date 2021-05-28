from typing import Dict, Any, Iterable, Tuple, List
from inspect import signature, getcallargs
from abc import abstractmethod, ABC
from copy import deepcopy
from packaging.version import parse

import torch
from torch import nn
from torch.nn import Module

from brevitas import nn as qnn
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.utils import merge_bn
from brevitas.fx import GraphModule, Graph, Node, map_arg
from brevitas.fx import immutable_list, immutable_dict, get_testing_overrides


_TORCH_TESTING_DICT = get_testing_overrides()


def _replace_all_users_except(to_replace: Node, replace_with: 'Node', exceptions=()):
    """
    Replace all users of ``to_replace`` with the Node ``replace_with``, except when
    the user is in exceptions.

    Args:
        to_replace (Node): The node to replace all uses of.
        replace_with (Node): The node to replace all uses of ``to_replace`` with.
        exceptions (List[Node]): The user nodes that should be affected.

    Returns:
        The list of Nodes on which this change was made.
    """
    to_process = list(to_replace.users)
    for use_node in to_process:
        def maybe_replace_node(n: Node) -> Node:
            if n == to_replace and use_node not in exceptions:
                return replace_with
            else:
                return n

        new_args = map_arg(use_node.args, maybe_replace_node)
        new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
        assert isinstance(new_args, tuple)
        assert isinstance(new_kwargs, dict)
        use_node._Node__update_args_kwargs(new_args, new_kwargs)
    return to_process


def _signature_keys(module_class):
    return signature(module_class).parameters.keys()


def _is_subseq(seq, subseq):
    return any(subseq == seq[i:len(subseq) + i] for i in range(len(seq) - len(subseq) + 1))


def _get_module_name_and_parent(model, fully_qualified_module_name):
    supermodule = model
    prefix_list = fully_qualified_module_name.split('.')
    module_name = prefix_list[-1]
    prefix_list = prefix_list[:-1]  # exclude module name
    for prefix in prefix_list:
        if prefix:  # exclude empty prefix
            supermodule = supermodule._modules[prefix]
    return module_name, supermodule


def _set_module(model, module, fully_qualified_module_name):
    module_name, supermodule = _get_module_name_and_parent(model, fully_qualified_module_name)
    supermodule._modules[module_name] = module


def _get_module(model, fully_qualified_module_name):
    named_modules = dict(model.named_modules())
    return named_modules[fully_qualified_module_name]


def _del_module(model, fully_qualified_module_name):
    module_name, supermodule = _get_module_name_and_parent(model, fully_qualified_module_name)
    del supermodule._modules[module_name]


def _name_from_module(model, module):
    for name, m in model.named_modules():
        if m is module:
            return name
    return None


def _replace_module(model, old_module, new_module):
    name = _name_from_module(model, old_module)
    _set_module(model, new_module, name)


# https://github.com/pytorch/pytorch/blob/v1.8.1/torch/fx/_experimental/fuser.py
# Works for length 2 patterns with 2 modules
def _matches_module_pattern(pattern: Iterable, node: Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


class Rewriter(ABC):

    @abstractmethod
    def apply(self, model: GraphModule) -> GraphModule:
        pass


class ModuleToModuleRewriter(Rewriter, ABC):

    def __init__(
            self,
            old_module_class,
            new_module_class,
            old_module_cond=lambda old_module: True,
            **kwargs):
        super().__init__()
        self.old_module_class = old_module_class
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs
        self.old_module_cond = old_module_cond

    def _map_origin_vars(self, vars: dict):
        return {k: v is not None if k == 'bias' else v for k, v in vars.items()}

    def _module_attributes(self, module):
        attrs = vars(module)
        # workaround since bias doesn't show up on vars of Linear
        if hasattr(module, 'bias'):
            attrs['bias'] = module.bias
        return attrs

    def _init_new_module(self, old_module: Module):
        # get attributes of original module
        new_kwargs = self._module_attributes(old_module)
        # transforms attribute of original module, e.g. bias Parameter -> bool
        new_kwargs = self._map_origin_vars(new_kwargs)
        # restrict to only values that are in the init of the new module
        new_module_signature_keys = _signature_keys(self.new_module_class)
        new_kwargs = {k: v for k, v in new_kwargs.items() if k in new_module_signature_keys}
        # update with kwargs passed to the rewriter
        new_kwargs.update(self.new_module_kwargs)
        # init the new module
        new_module = self.new_module_class(**new_kwargs)
        return new_module

    def _rewrite_model(self, model, old_new_module_dict):
        for old_module, new_module in old_new_module_dict.items():
            _replace_module(model, old_module, new_module)
            new_module.load_state_dict(old_module.state_dict())

    def apply(self, model: GraphModule):
        old_new_module_dict = {}
        for old_module in model.modules():
            # check for equality, not inheritance
            if type(old_module) == self.old_module_class and self.old_module_cond(old_module):
                # init the new module based on the old one
                new_module = self._init_new_module(old_module)
                # register modules pair to be replaced
                old_new_module_dict[old_module] = new_module
        # replace all pairs registered
        self._rewrite_model(model, old_new_module_dict)
        return model


class MergeBatchNorm2d(Rewriter):

    DEFAULT_PATTERNS = (
        (nn.Linear, nn.BatchNorm1d),
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (nn.ConvTranspose1d, nn.BatchNorm1d),
        (nn.ConvTranspose2d, nn.BatchNorm2d),
        (qnn.QuantLinear, nn.BatchNorm1d),
        (qnn.QuantConv1d, nn.BatchNorm1d),
        (qnn.QuantConv2d, nn.BatchNorm2d),
        (qnn.QuantConvTranspose1d, nn.BatchNorm1d),
        (qnn.QuantConvTranspose2d, nn.BatchNorm2d))

    def __init__(self, patterns=DEFAULT_PATTERNS):
        super(MergeBatchNorm2d, self).__init__()
        self.patterns = list(patterns)


    def apply(self, graph_model: GraphModule):
        named_modules = dict(graph_model.named_modules())
        for pattern in self.patterns:
            for node in graph_model.graph.nodes:
                if _matches_module_pattern(pattern, node, named_modules):
                    if len(node.args[0].users) > 1:  # Output of layer is used by other nodes
                        continue
                    layer = named_modules[node.args[0].target]
                    bn = named_modules[node.target]
                    merge_bn(layer, bn)
                    node.replace_all_uses_with(node.args[0])
                    graph_model.graph.erase_node(node)
                    _del_module(graph_model, node.target)
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class DuplicateSharedStatelessModule(Rewriter):

    def apply(self, graph_model: GraphModule):
        named_mods = graph_model.named_modules()  # duplicates are returned only once
        dup_mod_dict: Dict[str, int] = {}
        for name, mod in dict(named_mods).items():
            is_stateful = list(mod.parameters(recurse=True)) or list(mod.buffers(recurse=True))
            if not is_stateful:
                for node in list(graph_model.graph.nodes):
                    # duplicates are collapsed under the same target str during tracing
                    if isinstance(node.target, str) and node.target == name:
                        if name in dup_mod_dict.keys():
                            dup_mod_dict[name] += 1
                            dup_name = f'{name}_{dup_mod_dict[name]}'
                            _set_module(graph_model, deepcopy(mod), dup_name)
                            node.target = dup_name
                        else:
                            dup_mod_dict[name] = 0
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class DisableBreakingReturnQuantTensor(Rewriter):

    def is_breaking(self, graph_model, user):
        if (user.op == 'call_module'
                and hasattr(_get_module(graph_model, user.target), 'accept_quant_tensor')):
            return False
        if user.op == 'call_method' and hasattr(QuantTensor, user.target):
            return False
        if user.op == 'call_function' and hasattr(QuantTensor, f'__{user.target.__name__}__'):
            return False
        if user.op == 'call_function' and parse(torch.__version__) >= parse('1.5.0'):
            return False
        return True

    def apply(self, graph_model: GraphModule):
        for node in graph_model.graph.nodes:
            if node.op == 'call_module':
                module = _get_module(graph_model, node.target)
                if hasattr(module, 'return_quant_tensor') and module.return_quant_tensor:
                    for user in node.users:
                        if self.is_breaking(graph_model, user):
                            module.return_quant_tensor = False
        return graph_model


class CallableToModuleRewriter(Rewriter, ABC):

    def __init__(
            self,
            old_callable,
            new_module_class,
            **kwargs):
        super().__init__()
        self.old_callable = old_callable
        self.new_module_class = new_module_class
        self.new_module_kwargs = kwargs

    @abstractmethod
    def match_node(self, node: Node):
        pass

    def split_kwargs(self, node: Node):
        new_module_keys = _signature_keys(self.new_module_class)
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
            node.args = immutable_list(fn_args)
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
        _set_module(graph_model, module, module_name)

    def apply(self, graph_model: GraphModule) -> GraphModule:
        for node in graph_model.graph.nodes:
            if self.match_node(node):
                self.rewrite_node(node, graph_model)
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class FnToModuleRewriter(CallableToModuleRewriter):

    def match_node(self, node: Node) -> bool:
        return node.op == 'call_function' and node.target is self.old_callable


class MethodToModuleRewriter(CallableToModuleRewriter):

    def match_node(self, node: Node) -> bool:
        return node.op == 'call_method' and node.target is self.old_callable

    def move_node_args_to_kwargs(self, node: Node):
        if 'self' in node.kwargs:
            node_kwargs = dict(node.kwargs)
            slf = node_kwargs.pop('self')
            node.kwargs = immutable_dict(node_kwargs)
            node.args = immutable_list([slf] + list(node.args))


class MeanMethodToAdaptiveAvgPool2d(MethodToModuleRewriter):

    def __init__(self):
        super(MeanMethodToAdaptiveAvgPool2d, self).__init__(
            old_callable='mean',
            new_module_class=nn.AdaptiveAvgPool2d,
            output_size=(1, 1))

    def match_node(self, node: Node) -> bool:
        spr = super(MeanMethodToAdaptiveAvgPool2d, self).match_node(node)
        is_adaptive_2d_mean = (
                (2, 3) in node.args
                or [2, 3] in node.args
                or 'dim' in node.kwargs
                and (node.kwargs['dim'] == (2, 3) or node.kwargs['dim'] == [2, 3]))
        return spr and is_adaptive_2d_mean

    def move_node_args_to_kwargs(self, node: Node):
        if 'dim' in node.kwargs:
            node.kwargs = immutable_dict(dict(node.kwargs).pop('dim'))
        elif (2, 3) in node.args or [2, 3] in node.args:
            node.args = immutable_list([a for a in node.args if a != (2, 3) and a != [2, 3]])

    def rewrite_node(self, node: Node, graph_model: GraphModule):
        super(MeanMethodToAdaptiveAvgPool2d, self).rewrite_node(node, graph_model)
        # the output of AdaptiveAvgPool2d is 4d, we need to squeeze it to match mean
        with graph_model.graph.inserting_after(node):
            batch_size_node = graph_model.graph.call_method('size', args=(node, 0))
        with graph_model.graph.inserting_after(batch_size_node):
            squeeze_node = graph_model.graph.call_method(
                'reshape', args=(node, (batch_size_node, -1)))
        _replace_all_users_except(node, squeeze_node, [squeeze_node, batch_size_node])


class AllignScaling(Rewriter):

    def apply(self, model: GraphModule) -> GraphModule:
        pass


class RewriterList(Rewriter):

    def __init__(self, rewriter_list):
        self.rewriter_list = rewriter_list

    def apply(self, model: GraphModule):
        for rewriter in self.rewriter_list:
            rewriter.apply(model)
        return model