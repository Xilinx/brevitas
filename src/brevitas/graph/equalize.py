import copy
from typing import Dict, Any
import operator
from random import sample
from functools import partial

import torch

from brevitas.fx import GraphModule, Node
from brevitas.graph.utils import get_module
from .base import GraphTransform

__all__ = [
    'EqualizeGraph'
]

EPSILON = 1e-9

_supported_layers = (
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.Linear)

_scale_invariant_layers = (
    torch.nn.Dropout,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.ReLU,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d)

_residual_methods = (
    'add',
    'add_'
)

_residual_fns = (
    torch.add,
    operator.add,
    operator.iadd,
    operator.__add__,
    operator.__iadd__
)

_batch_norm = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
)


def _channel_range(inp):
    mins, _ = inp.min(dim=1) 
    maxs, _ = inp.max(dim=1) 
    out = maxs - mins
    # correct corner case where where all weights along a channel have the same value
    # e.g. when a mean/nn.AvgPool/nn.AdaptiveAvgPool is converted to a depth-wise conv
    out = torch.where(out == 0., torch.mean(inp, dim=1), out)
    return out


def _get_size(axes):
    m0, axis0 = list(axes.items())[0]
    size = m0.weight.size(axis0)
    for m, axis in axes.items():
        if m.weight.size(axis) != size:
            raise RuntimeError("Weights sizes don't match")
    return size


def _get_input_axis(module):
    if isinstance(module, torch.nn.Linear):
        return 1
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        if module.groups == 1:
            return 1
        elif module.groups == module.out_channels:
            return 0
        else:
            raise RuntimeError("Group convolution not supported")
    elif isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
        if module.groups == 1:
            return 0
        elif module.groups == module.out_channels:
            return 1
        else:
            raise RuntimeError("Group convolution not supported")
    else:
        raise RuntimeError(f"Module {module} not supported.")


def _get_output_axis(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return 0
    elif isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranpose3d)):
        return 1
    else:
        raise RuntimeError(f"Module {module} not supported.")


def _cross_layer_equalization(srcs, sinks):
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    for module_set in [srcs, sinks]:
        for module in module_set:
            if not isinstance(module, _supported_layers):
                raise ValueError("module type not supported:", type(module))

    src_axes = {m: _get_output_axis(m) for m in srcs}
    sink_axes = {m: _get_input_axis(m) for m in sinks}
    src_size = _get_size(src_axes)
    sink_size = _get_size(sink_axes)

    if src_size != sink_size:
        raise RuntimeError("Output channels of sources do not match input channels of sinks")

    transpose = lambda module, axis: module.weight if axis == 0 else module.weight.transpose(0, 1)
    src_weights = [transpose(m, axis) for m, axis in src_axes.items()]
    sink_weights = [transpose(m, axis) for m, axis in sink_axes.items()]
    srcs_range = _channel_range(torch.cat([w.reshape(w.size(0), -1) for w in src_weights], 1))
    sinks_range = _channel_range(torch.cat([w.reshape(w.size(0), -1) for w in sink_weights], 1))
    sinks_range += EPSILON
    scaling_factors = torch.sqrt(srcs_range / sinks_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    for module, axis in src_axes.items():
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data * inverse_scaling_factors.view_as(module.bias)
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        module.weight.data = module.weight.data * torch.reshape(inverse_scaling_factors, src_broadcast_size)
    for module, axis in sink_axes.items():
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        module.weight.data = module.weight.data * torch.reshape(scaling_factors, src_broadcast_size)


def _equalize(model, regions, iterations):
    """
    Generalized version of section 4.1 of https://arxiv.org/pdf/1906.04721.pdf
    """
    name_to_module : Dict[str, torch.nn.Module] = {}
    name_set = {name for region in regions for module_set in region for name in module_set}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    for i in range(iterations):
        for region in regions:
            _cross_layer_equalization([name_to_module[n] for n in region[0]], [name_to_module[n] for n in region[1]])
    return model


def _is_supported_module(graph_model, node):
    return node.op == 'call_module' and isinstance(get_module(graph_model, node.target), _supported_layers)


def _is_scale_invariant_module(graph_model, node):
    return node.op == 'call_module' and isinstance(get_module(graph_model, node.target), _scale_invariant_layers)


def _is_reshaping_op(node):
    return (node.op == 'call_function' and node.target in [torch.flatten, torch.reshape]
                or node.op == 'call_method' and node.target in ['view', 'reshape', 'flatten'])


def walk_region(graph_model: GraphModule, starting_node: Node, history, srcs, sinks, walk_forward):
    node_list = starting_node.users if walk_forward else starting_node.all_input_nodes
    for node in node_list:
        # we keep a history of how the graph has been walked already, invariant to the direction,
        # to avoid getting stuck in a loop
        path = (starting_node, node) if walk_forward else (node, starting_node)
        if path not in history:
            history.add(path)
        else:
            continue
        if _is_supported_module(graph_model, node):
            if walk_forward:
                sinks.add(node.target)
            else:
                srcs.add(node.target)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
        elif _is_scale_invariant_module(graph_model, node):
            if walk_forward:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
            else:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        elif (node.op == 'call_method' and node.target in _residual_methods
            or node.op == 'call_function' and node.target in _residual_fns):
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        elif _is_reshaping_op(node):
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=walk_forward)
        else:
            continue


def _extract_regions(graph_model: GraphModule):
    regions = set()
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if isinstance(module, _supported_layers):
                srcs, sinks = {node.target}, set()
                walk_region(graph_model, node, set(), srcs, sinks, walk_forward=True)
                if sinks:
                    # each region should appear only once, so to make it hashable
                    # we convert srcs and sinks to ordered lists first, and then to tuples
                    regions.add((tuple(sorted(srcs)), tuple(sorted(sinks))))
    # for clarity, sort by the of the first source
    regions = sorted(regions, key=lambda region: region[0][0])
    return regions


class EqualizeGraph(GraphTransform):

    def __init__(self, iterations) -> None:
        super(EqualizeGraph, self).__init__()
        self.iterations = iterations 

    def apply(self, graph_model: GraphModule):
        regions = _extract_regions(graph_model)
        graph_model = _equalize(graph_model, regions, self.iterations)
        return graph_model


class AbsorbBiasByBatchNorm(GraphTransform):

    def __init__(self):
        super(AbsorbBiasByBatchNorm, self).__init__()
        self.inp_shape_map = {}
        self.collect_inp_shape_hooks = []

    def add_to_bias(self, module, tensor):
        if module.bias is not None:
            module.bias.data += tensor.view_as(module.bias)
        else:
            module.bias = torch.nn.Parameter(tensor)

    def absorb_biases(self, groups):
        for layer, bn, (next_layer_name, next_layer) in groups:
            cfactor = bn.running_mean - 3 * torch.sqrt(bn.running_var)
            zeroes = torch.zeros_like(cfactor).to(cfactor.device)
            cfactor = torch.where(cfactor > 0., cfactor, zeroes)
            if (cfactor > 0).any():
                self.add_to_bias(layer, -cfactor)
                broadcast_shape = [1] * next_layer.weight.ndim
                broadcast_shape[1] = cfactor.numel()
                cfactor = cfactor.view(broadcast_shape)
                cfactor = cfactor.expand(self.inp_shape_map[next_layer_name])
                next_layer_cfactor = next_layer(cfactor).transpose(0, 1)
                next_layer_cfactor = next_layer_cfactor.view(next_layer_cfactor.shape[0], -1)
                next_layer_cfactor = torch.mean(next_layer_cfactor, dim=1)
                self.add_to_bias(next_layer, next_layer_cfactor)
        self.inp_shape_map = {}

    def extract_groups(self, graph_model: GraphModule):
        groups = []
        for node in graph_model.graph.nodes:
            if (_is_supported_module(graph_model, node)
                and node.next.op == 'call_module'
                and isinstance(get_module(graph_model, node.next.target), _batch_norm)):
                node_next = node.next.next
                while _is_scale_invariant_module(graph_model, node_next) or _is_reshaping_op(node_next):
                    node_next = node_next.next
                if _is_supported_module(graph_model, node_next):
                    group = (
                        get_module(graph_model, node.target), 
                        get_module(graph_model, node.next.target), 
                        (node_next.target, get_module(graph_model, node_next.target)))
                    groups.append(group)
        return groups

    def collect_inp_shape_hook(self, module, inp, name):
        if name in self.inp_shape_map.keys():
            raise RuntimeError("Module called multiple times, not supported.")
        if isinstance(inp, tuple):
            inp = inp[0]
        self.inp_shape_map[name] = [1] + list(inp.shape[1:])

    def collect_inp_shapes(self, model, inp):
        for name, module in model.named_modules():
            if isinstance(module, (_supported_layers)):
                hook_fn = partial(self.collect_inp_shape_hook, name=name)
                hook = module.register_forward_pre_hook(hook_fn)
                self.collect_inp_shape_hooks.append(hook)
        model(inp)
        for hook in self.collect_inp_shape_hooks:
            hook.remove()
        self.collect_inp_shape_hooks = []

    def apply(self, graph_model: GraphModule, inp):
        self.collect_inp_shapes(graph_model, inp)
        groups = self.extract_groups(graph_model)
        self.absorb_biases(groups)
        return graph_model