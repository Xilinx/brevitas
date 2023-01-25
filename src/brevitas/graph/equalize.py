# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from collections import OrderedDict
from copy import deepcopy
from functools import partial
import operator
from typing import Dict, List, Union

import torch

from brevitas.fx import GraphModule
from brevitas.fx import Node
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
    torch.nn.Linear,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,)

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
    operator.__iadd__,
)

_cat_fns = (torch.cat,)

_batch_norm = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
)


def _channel_range(module: torch.nn.Module):
    mins, _ = module.min(dim=1)
    maxs, _ = module.max(dim=1)
    out = maxs - mins
    # correct corner case where where all weights along a channel have the same value
    # e.g. when a mean/nn.AvgPool/nn.AdaptiveAvgPool is converted to a depth-wise conv
    out = torch.where(out == 0., torch.mean(module, dim=1), out)
    return out


def _get_size(axes: Dict[torch.nn.Module, int], check_same: bool =False):
    """
    Return the sizes of the nn.Modules in axes. If check_same is set to True, it fails if there
    are different sizes along the module-specific axis.
    """
    m0, axis0 = list(axes.items())[0]
    size = m0.weight.size(axis0)
    sizes = torch.zeros(len(list(axes.keys())))
    for i, (m, axis) in enumerate(axes.items()):
        sizes[i] = m.weight.size(axis)
        if check_same and sizes[i] != size:
            raise RuntimeError("Weights sizes don't match")
    return sizes


def _get_input_axis(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        return 1
    elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        return 0
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


def _get_output_axis(module: torch.nn.Module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        return 0
    elif isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranpose3d)):
        return 1
    else:
        raise RuntimeError(f"Module {module} not supported.")


def _combine_weights_bias(module: torch.nn.Module, axis: int, bias_shrinkage: Union[int, float, str]):
    """Combine weights and bias before graph equalizattion

    This method merges the weight and bias sources, so that the resulting equalizer scale factor
    is influenced also by the magnitude of the bias, the latter mitigated by a shrink factor.
    This technique avoids that, after the equalization procedure, the bias values become too big,
    negatively impacting the quantization accuracy.
    The bias shrinkage factor regulates how much the bias magnitude will affect the subsequence
    calculation of the scale.
    """
    if module.bias is None:
        return module.weight.data

    bias = module.bias.data
    weight = module.weight.data if axis == 0 else module.weight.data.transpose(1,0)
    weight = deepcopy(weight).view(weight.shape[0], -1)
    bias = deepcopy(bias).view(-1, 1)

    weight = torch.where(torch.abs(weight) < EPSILON, torch.tensor(EPSILON).type_as(weight), weight)
    factor = torch.abs(bias)/torch.abs(weight)

    # From https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/nndct_shared/optimization/commander.py#L450
    if bias_shrinkage == 'vaiq':
        if torch.abs(bias).max() < 10 and (torch.abs(bias).max()/torch.abs(weight).max() ) < 20:
            if torch.median(factor) > 100 or torch.mean(factor) > 1000:
                shrink_factor = 5
            else:
                shrink_factor = 2
        else:
            if  torch.median(factor) > 30 or torch.mean(factor) > 500:
                shrink_factor = 20
            elif torch.median(factor) > 15 or torch.mean(factor) > 250:
                shrink_factor = 10
            else:
                shrink_factor = 5
    elif isinstance(bias_shrinkage, (int,float)):
        shrink_factor = bias_shrinkage
    else:
        raise RuntimeError(f"{bias_shrinkage} not supported.")
    weight_bias = torch.cat([weight, bias/shrink_factor], 1)
    return weight_bias

def _cross_layer_equalization(srcs: List, sinks: List, merge_bias: bool, bias_shrinkage: Union[int, float, str]):
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    for module_set in [srcs, sinks]:
        for module in module_set:
            if not isinstance(module, _supported_layers):
                raise ValueError("module type not supported:", type(module))

    # Because of possible concat, we need to preserve the order
    src_axes = OrderedDict([(m, _get_output_axis(m)) for m in srcs])
    sink_axes = OrderedDict([(m, _get_input_axis(m)) for m in sinks])
    src_sink = _get_size(src_axes)
    sink_size = _get_size(sink_axes, check_same=True)

    if torch.sum(src_sink) == sink_size[0] and len(src_sink)>1:
        is_concat = True
    elif torch.mean(src_sink) == sink_size[0]: # Workaround to check src_sink have all the same size
        is_concat = False
    else:
        # Return without equalizing
        return torch.ones(1)

    transpose = lambda module, axis: module.weight if axis == 0 else module.weight.transpose(0, 1)
    if merge_bias:
        src_weights = [_combine_weights_bias(m, axis, bias_shrinkage) for m, axis in src_axes.items()]
    else:
        src_weights = [transpose(m, axis) for m, axis in src_axes.items()]
    sink_weights = [transpose(m, axis) for m, axis in sink_axes.items()]
    if is_concat:
        srcs_range = torch.cat([_channel_range(w.reshape(w.size(0), -1))  for w in src_weights],0)
    else:
        srcs_range = _channel_range(torch.cat([w.reshape(w.size(0), -1) for w in src_weights], 1))
    sinks_range = _channel_range(torch.cat([w.reshape(w.size(0), -1) for w in sink_weights], 1))
    sinks_range += EPSILON

    # If the ratio is negative, set the scale to 1 (i.e., no-op)
    scaling_factors = torch.where((srcs_range/sinks_range)>0, torch.sqrt(srcs_range / sinks_range), \
        torch.tensor(1.).type_as(srcs_range))

    start_index = 0
    inverse_scaling_factors = torch.reciprocal(scaling_factors)
    for module, axis in src_axes.items():
        channels = module.weight.size(axis)
        partial_inverse_scaling = inverse_scaling_factors[start_index:start_index+channels]
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data * partial_inverse_scaling.view_as(module.bias)
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        module.weight.data = module.weight.data * torch.reshape(partial_inverse_scaling, src_broadcast_size)
        if is_concat:
            start_index += channels
    for module, axis in sink_axes.items():
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        module.weight.data = module.weight.data * torch.reshape(scaling_factors, src_broadcast_size)
    return scaling_factors


def _equalize(model: GraphModule, regions: tuple, iterations: int, threshold: Union[float, None], merge_bias: bool, bias_shrinkage: Union[int, float, str]):
    """
    Generalized version of section 4.1 of https://arxiv.org/pdf/1906.04721.pdf
    """
    name_to_module : Dict[str, torch.nn.Module] = {}
    # name_set = {name for region in regions for module_set in region for name in module_set}
    name_set = set()
    for region in regions:
        for src_reg in region[0]:
            name_set.add(src_reg[0])
        name_set.update(region[1])

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    for i in range(iterations):
        scale_factor_max = None
        for region in regions:

            # Sources are re-ordered, necessary for equalizing through cat
            srcs_ordered = [0] * len(region[0])
            for v, k in region[0]:
                srcs_ordered[k] = v

            scale_factors_region = _cross_layer_equalization([name_to_module[n] for n in srcs_ordered], [name_to_module[n] for n in region[1]], merge_bias, bias_shrinkage)
            scale_factor_region_max = torch.max(torch.abs(1-scale_factors_region))
            scale_factor_max = torch.max(scale_factor_max, scale_factor_region_max) if \
                scale_factor_max is not None else scale_factor_region_max
        if threshold is not None:
            if scale_factor_max < threshold:
                break
    return model


def _is_supported_module(graph_model: GraphModule, node: Node):
    return node.op == 'call_module' and isinstance(get_module(graph_model, node.target), _supported_layers)


def _is_scale_invariant_module(graph_model: GraphModule, node: Node):
    return node.op == 'call_module' and isinstance(get_module(graph_model, node.target), _scale_invariant_layers)


def _is_reshaping_op(node: Node):
    return (node.op == 'call_function' and node.target in [torch.flatten, torch.reshape]
                or node.op == 'call_method' and node.target in ['view', 'reshape', 'flatten'])

def _check_nodes(srcs: Dict, name: str, node: Node, current_idx: List, graph_model: GraphModule):
    """
    Recursive search up the graph, starting from node, until it finds the target node with given
    name, and updates the srcs dict with the correct index position.
    """
    if _is_supported_module(graph_model, node):
        if name in srcs and node.target == name:
            srcs[name] = current_idx[0]
            current_idx[0] += 1
            return True
    else:
        # Check if it's a node through which we would propagate in walk_region
        # Otherwise the source cannot come from here, so we stop the search
        if _is_scale_invariant_module(graph_model, node) or \
        (node.op == 'call_method' and node.target in _residual_methods
            or node.op == 'call_function' and node.target in _residual_fns) or \
                node.target in _cat_fns or _is_reshaping_op(node):
            for input_node in node.all_input_nodes:
                done = _check_nodes(srcs, name, input_node, current_idx, graph_model)
                if done:
                    return done
        return False


def _reorder_sources(srcs: Dict, node: Node, graph_model: GraphModule):
    """Re-order sources of node
    This function reorders the sources (srcs) of a specific node.
    When performing Graph Equalization, sources can be discovered and stored in any order.
    When equalizing through certain operators like concatenation, it is fundamental to reconstruct
    the original order, so that the channel of the sources are correctly aligned with the ones of
    the sinks.

    """
    all_input_nodes = node.all_input_nodes
    current_idx = [0]
    srcs_found = 0
    list_to_check = list(srcs.keys())
    while srcs_found < len(srcs.keys()):
        for inner_node in all_input_nodes:
            done = False
            for name in list_to_check:
                done = _check_nodes(srcs, name, inner_node, current_idx, graph_model)
                if done:
                    break
            if done:
                srcs_found +=1
                list_to_check.remove(name)

def _clear_sinks(sinks):
    """
    Delete all the sinks
    """
    all_sinks = list(sinks.keys())
    for keys in all_sinks:
        del sinks[keys]

def walk_region(graph_model: GraphModule, starting_node: Node, history, srcs: dict, sinks: dict, walk_forward: bool, seen_concat: bool = False):
    """
    Recursive algorithm to walk through the graph, to detect all possible sources (srcs) and sinks
    that can be equalized together.
    This method is able to correctly idefinity regions across residual connection, batch norms,
    concatenations, and scale-invariant and operators such as AveragePool, MaxPool or ReLU.
    """
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
            # If we find a supported module walking forward, it is a sink, otherwise it is a source.
            # This can happen when meeting residual or cat nodes.
            if walk_forward:
                if node.target not in sinks:
                    sinks[node.target] = 0
            else:
                if node.target not in srcs:
                    srcs[node.target] = len(srcs.keys())
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True, seen_concat=seen_concat)
        elif _is_scale_invariant_module(graph_model, node):
            if walk_forward:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True, seen_concat=seen_concat)
            else:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True, seen_concat=seen_concat)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False, seen_concat=seen_concat)
        elif (node.op == 'call_method' and node.target in _residual_methods
            or node.op == 'call_function' and node.target in _residual_fns):
                # If we see a residual add, we need to walk backward to find the other sources to
                # the add, as well as forward to register the sinks.
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True, seen_concat=seen_concat)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False, seen_concat=seen_concat)
        elif node.target in _cat_fns:
            if seen_concat:
                # If we go through another concat, then no equalization should be done.
                # Removing all the sinks should guarantee that.
                _clear_sinks(sinks)
                continue

            # If any of the input to cat has multiple users, we cannot equalize through
            condition = [len(n.users)>1 for n in node.all_input_nodes]
            if any(condition):
                continue

            # If this is the first concat we see, we register it while we keep exploring the graph.
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=True, seen_concat=True)
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=False, seen_concat=True)

            # If we have sinks at the end of our search, we have a valid region thus we reorder the
            # sources
            if len(sinks) > 0:
                _reorder_sources(srcs, node, graph_model)
        elif _is_reshaping_op(node):
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=walk_forward, seen_concat=seen_concat)
        else:
            continue


def _extract_regions(graph_model: GraphModule):
    """Find potential regions for Graph Equalization
    This methods iterates through all possible nodes and define a unique set of regions that will
    be used for graph equalization.
    Each region is composed by one or multiple sources, starting from the selected node, and one or
    multiple sinks. If no sinks are found, then no region is added, and the search moves on to the
    next node.
    """
    regions = set()
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if isinstance(module, _supported_layers):
                srcs = {node.target: 0}
                sinks = dict()
                walk_region(graph_model, node, set(), srcs, sinks, walk_forward=True)
                if sinks:
                    # each region should appear only once, so we convert to sorted tuples
                    srcs = [(x, pos) for x, pos in srcs.items()]
                    # srcs_ordered = [0] * len(srcs.items())
                    # for v, k in srcs.items():
                    #     srcs_ordered[k] = v
                    regions.add((tuple(sorted(srcs)), tuple(sorted(sinks.keys()))))
    # for clarity, sort by the of the first source
    regions = sorted(regions, key=lambda region: region[0][0])
    return regions



class EqualizeGraph(GraphTransform):
    """
    Graph transformation that equalizes weights across scale invariant operators.
    For more information, check https://arxiv.org/pdf/1906.04721.pdf

    Args:
        iterations (int): Number of iterations for equalization. Defaults to 10.
        threshold (float): Minimum value of scale factor changes across regions, under which the
            equalization procedure is terminated early. Defaults to 0.05.
        merge_bias (bool): Whether to merge bias into the weight of the source modules when
            computing the equalizing scale factors.
        bias_shrinkage (int, float, str): A number indicating the shrink factor for when merging
            bias and weights, or pass 'vaiq' to use the heuristic defined by Vitis-AI quantizer.
            Ignored if merge_bias=False.
    """
    def __init__(self, iterations=10, threshold=0.05, merge_bias=True, bias_shrinkage='vaiq') -> None:
        super(EqualizeGraph, self).__init__()
        self.iterations = iterations
        self.threshold = threshold
        self.merge_bias = merge_bias
        self.bias_shrinkage = bias_shrinkage

    def apply(self, graph_model: GraphModule):
        regions = _extract_regions(graph_model)
        graph_model = _equalize(graph_model, regions, self.iterations, self.threshold, self.merge_bias, self.bias_shrinkage)
        return graph_model, regions


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
