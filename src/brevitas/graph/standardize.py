from copy import deepcopy
from typing import Dict

from torch import nn
import torch.nn.functional as F

from brevitas.fx import GraphModule, Node, immutable_dict
from .base import GraphTransform, MethodToModule, FnToModule
from .utils import set_module, replace_all_uses_except, get_module

__all__ = [
    'DuplicateSharedStatelessModule',
    'MeanMethodToAdaptiveAvgPool2d',
    'TorchFunctionalToModule',
    'DisableLastReturnQuantTensor'
]


class DuplicateSharedStatelessModule(GraphTransform):

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
                            set_module(graph_model, deepcopy(mod), dup_name)
                            node.target = dup_name
                        else:
                            dup_mod_dict[name] = 0
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class MeanMethodToAdaptiveAvgPool2d(MethodToModule):

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
            node.args = tuple([a for a in node.args if a != (2, 3) and a != [2, 3]])

    def rewrite_node(self, node: Node, graph_model: GraphModule):
        super(MeanMethodToAdaptiveAvgPool2d, self).rewrite_node(node, graph_model)
        # the output of AdaptiveAvgPool2d is 4d, we need to squeeze it to match mean
        with graph_model.graph.inserting_after(node):
            batch_size_node = graph_model.graph.call_method('size', args=(node, 0))
        with graph_model.graph.inserting_after(batch_size_node):
            squeeze_node = graph_model.graph.call_method(
                'reshape', args=(node, (batch_size_node, -1)))
        replace_all_uses_except(node, squeeze_node, [squeeze_node, batch_size_node])


class TorchFunctionalToModule(GraphTransform):

    FN_TO_MODULE_MAP = (
        (F.relu, nn.ReLU),
        (F.relu_, nn.ReLU),
        (F.relu6, nn.ReLU6),
        (F.hardtanh, nn.Hardtanh),
        (F.hardtanh_, nn.Hardtanh),
        (F.leaky_relu, nn.LeakyReLU),
        (F.leaky_relu_, nn.LeakyReLU),
        (F.max_pool1d, nn.MaxPool1d),
        (F.max_pool2d, nn.MaxPool2d),
        (F.max_pool3d, nn.MaxPool3d),
        (F.avg_pool1d, nn.AvgPool1d),
        (F.avg_pool2d, nn.AvgPool2d),
        (F.avg_pool3d, nn.AvgPool3d),
        (F.adaptive_avg_pool1d, nn.AdaptiveAvgPool1d),
        (F.adaptive_avg_pool2d, nn.AdaptiveAvgPool2d),
        (F.adaptive_avg_pool3d, nn.AdaptiveAvgPool3d))

    def __init__(self, fn_to_module_map=FN_TO_MODULE_MAP):
        super().__init__()
        self.rewriter_list = [FnToModule(fn, mclass) for (fn, mclass) in fn_to_module_map]

    def apply(self, model: GraphModule) -> GraphModule:
        for rewriter in self.rewriter_list:
            model = rewriter.apply(model)
        return model


class DisableLastReturnQuantTensor(GraphTransform):

    def apply(self, graph_model: GraphModule):
        for node in graph_model.graph.nodes:
            if node.op == 'call_module':
                module = get_module(graph_model, node.target)
                if hasattr(module, 'return_quant_tensor') and module.return_quant_tensor:
                    if len(node.users) == 1 and list(node.users)[0].op == 'output':
                        module.return_quant_tensor = False
        return graph_model