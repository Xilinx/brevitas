import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.fx import GraphModule, immutable_dict, immutable_list
from brevitas.nn.utils import merge_bn
from .base import UntilFixedPointGraphTransform
from .utils import del_module
from .utils import replace_all_uses_except
from .utils import get_module
from .utils import get_output_channels
from .utils import get_output_channel_dim
from .utils import matches_module_pattern

__all__ = [
    'MoveSplitBatchNormBeforeCat',
    'MergeBatchNorm',
    'CollapseConsecutiveConcats'
]


class MoveSplitBatchNormBeforeCat(UntilFixedPointGraphTransform):

    DEFAULT_BEFORE_MODULES_TYPES = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        qnn.QuantLinear,
        qnn.QuantConv1d,
        qnn.QuantConv2d,
        qnn.QuantConvTranspose1d,
        qnn.QuantConvTranspose2d)

    def __init__(self, before_modules_types=DEFAULT_BEFORE_MODULES_TYPES):
        super(MoveSplitBatchNormBeforeCat, self).__init__()
        self.before_modules_types = before_modules_types

    def is_converged(self, graph_model: GraphModule) -> bool:
        for cat_node in graph_model.graph.nodes:
            if (cat_node.target is torch.cat
                    and len(cat_node.users) == 1
                    and cat_node.kwargs['dim'] == 1
                    and list(cat_node.users)[0].op == 'call_module'):
                bn_node = list(cat_node.users)[0]
                module = get_module(graph_model, bn_node.target)
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    inp_nodes = cat_node.all_input_nodes
                    if all([inp_node.op == 'call_module' 
                        and len(inp_node.users) == 1 for inp_node in inp_nodes]):
                        before_mods = [
                            get_module(graph_model, inp_node.target) for inp_node in inp_nodes]
                        if all(isinstance(mod, self.before_modules_types) for mod in before_mods):
                            assert inp_nodes == cat_node.kwargs['tensors']
                            num_features_list = [get_output_channels(mod) for mod in before_mods]
                            chunk_bn_list = [type(module)(n) for n in num_features_list]
                            for i, chunk_bn in enumerate(chunk_bn_list):
                                chunk_bn_name = f'{bn_node.name}_{i}'
                                graph_model.add_module(chunk_bn_name, chunk_bn)
                                start = sum(num_features_list[:i])
                                end = sum(num_features_list[:i+1])
                                chunk_bn.weight.data = module.weight.data[start:end]
                                chunk_bn.bias.data = module.bias.data[start:end]
                                chunk_bn.running_mean = module.running_mean.data[start:end]
                                chunk_bn.running_var = module.running_var.data[start:end]
                                inp_node = cat_node.kwargs['tensors'][i]
                                with graph_model.graph.inserting_after(inp_node):
                                    chunk_bn_node = graph_model.graph.call_module(
                                        chunk_bn_name, args=(inp_node,))
                                replace_all_uses_except(inp_node, chunk_bn_node, [chunk_bn_node])
                            bn_node.replace_all_uses_with(cat_node)
                            graph_model.graph.erase_node(bn_node)
                            del_module(graph_model, bn_node.target)
                            graph_model.graph.lint()
                            graph_model.recompile()
                            return False
        return True


class MergeBatchNorm(UntilFixedPointGraphTransform):

    DEFAULT_PATTERNS = (
        (nn.BatchNorm1d, nn.BatchNorm1d),
        (nn.BatchNorm2d, nn.BatchNorm2d),
        (nn.BatchNorm3d, nn.BatchNorm3d),
        (nn.Linear, nn.BatchNorm1d),
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (nn.ConvTranspose1d, nn.BatchNorm1d),
        (nn.ConvTranspose2d, nn.BatchNorm2d),
        (nn.ConvTranspose3d, nn.BatchNorm3d),
        (qnn.BatchNorm1dToQuantScaleBias, nn.BatchNorm1d),
        (qnn.BatchNorm2dToQuantScaleBias, nn.BatchNorm2d),
        (qnn.QuantLinear, nn.BatchNorm1d),
        (qnn.QuantConv1d, nn.BatchNorm1d),
        (qnn.QuantConv2d, nn.BatchNorm2d),
        (qnn.QuantConvTranspose1d, nn.BatchNorm1d),
        (qnn.QuantConvTranspose2d, nn.BatchNorm2d))

    def __init__(self, patterns=DEFAULT_PATTERNS):
        super(MergeBatchNorm, self).__init__()
        self.patterns = list(patterns)

    def is_converged(self, graph_model: GraphModule):
        named_modules = dict(graph_model.named_modules())
        for node in graph_model.graph.nodes:
            for pattern in self.patterns:
                if matches_module_pattern(pattern, node, named_modules):
                    if len(node.args[0].users) > 1:
                        continue
                    layer = named_modules[node.args[0].target]
                    bn = named_modules[node.target]
                    merge_bn(layer, bn, get_output_channel_dim(layer))
                    node.replace_all_uses_with(node.args[0])
                    graph_model.graph.erase_node(node)
                    del_module(graph_model, node.target)
        graph_model.recompile()
        graph_model.graph.lint()
        return graph_model


class CollapseConsecutiveConcats(UntilFixedPointGraphTransform):

    def merge_tensor_args(self, node_to_extract, node_to_merge_in):
        cat_tensors1 = list(node_to_extract.args[0])
        cat_tensors2 = node_to_merge_in.args[0]
        if not isinstance(cat_tensors2, (list, tuple)):
            cat_tensors2 = [cat_tensors2]
        cat_tensors2 = [t for t in cat_tensors2 if t is not node_to_extract]
        cat_tensors = immutable_list(cat_tensors1 + cat_tensors2)
        node_to_merge_in.args = (cat_tensors,)

    def is_converged(self, graph_model):
        for i, node in enumerate(graph_model.graph.nodes):
            if node.op == 'call_function' and node.target is torch.cat:
                for inp_node in node.all_input_nodes:
                    if (inp_node.op == 'call_function'
                            and inp_node.target is torch.cat
                            and node.kwargs['dim'] == inp_node.kwargs['dim']
                            and len(inp_node.users) == 1):
                        self.merge_tensor_args(inp_node, node)
                        graph_model.graph.erase_node(inp_node)
                        graph_model.graph.lint()
                        graph_model.recompile()
                        return False
        return True

    def move_args_to_kwargs(self, graph_model):
        for node in graph_model.graph.nodes:
            if node.op == 'call_function' and node.target is torch.cat:
                if len(node.args) > 0:
                    if isinstance(node.args[-1], int):
                        kwargs = dict(node.kwargs)
                        kwargs['dim'] = node.args[-1]
                        node.kwargs = immutable_dict(kwargs)
                        node.args = node.args[:-1]
                    if isinstance(node.args[0], (tuple, list)):
                        kwargs = dict(node.kwargs)
                        kwargs['tensors'] = node.args[0]
                        node.kwargs = immutable_dict(kwargs)
                        node.args = node.args[1:]

    def apply(self, graph_model: GraphModule) -> GraphModule:
        self.move_args_to_kwargs(graph_model)
        return super(CollapseConsecutiveConcats, self).apply(graph_model)
