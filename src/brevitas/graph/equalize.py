import copy
from typing import Dict, Any
import operator
from random import sample

import torch

from brevitas.fx import GraphModule, Node
from brevitas.graph.utils import get_module


EPSILON = 1e-9

# layer: (input_axis, output_axis)
_supported_layers = {
    torch.nn.ConvTranspose1d: (0, 1),
    torch.nn.ConvTranspose2d: (0, 1),
    torch.nn.Conv1d: (1, 0),
    torch.nn.Conv2d: (1, 0),
    torch.nn.Linear: (1, 0)}

_scale_invariant_layers = (
    torch.nn.Dropout,
    torch.nn.Dropout2d,
    torch.nn.ReLU,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d)

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


def _channel_range(inp):
    """ finds the range of weights associated with a specific channel"""
    mins, _ = inp.min(dim=1) # min_over_ndim(input, axis_list)
    maxs, _ = inp.max(dim=1) # max_over_ndim(input, axis_list)
    return maxs - mins


def _get_size(axes):
    m0, axis0 = list(axes.items())[0]
    size = m0.weight.size(axis0)
    for m, axis in axes.items():
        if m.weight.size(axis) != size:
            raise RuntimeError("Source weights do not have the same output size")
    return size


def _cross_layer_equalization(srcs, sinks):
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    for module_set in [srcs, sinks]:
        for module in module_set:
            if not isinstance(module, tuple(_supported_layers.keys())):
                raise ValueError("module type not supported:", type(module))

    src_axes = {m: v[1] for m in srcs for k, v in _supported_layers.items() if isinstance(m, k)}
    sink_axes = {m: v[0] for m in sinks for k, v in _supported_layers.items() if isinstance(m, k)}
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


def _norm(curr_modules, prev_modules):
    """
    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold
    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary
    """
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError("Keys to the models dicts do not match.")

    summed_norms = torch.tensor(0.)
    if None in prev_modules.values():
        return None
    for name in curr_modules.keys():
        difference = curr_modules[name].weight.sub(prev_modules[name].weight)
        summed_norms += torch.norm(difference)
    return summed_norms


def _equalize(model, regions, threshold, max_iterations):
    """
    Generalized version of section 4.1 of https://arxiv.org/pdf/1906.04721.pdf
    """
    name_to_module : Dict[str, torch.nn.Module] = {}
    previous_name_to_module: Dict[str, Any] = {}
    name_set = {name for region in regions for module_set in region for name in module_set}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    iteration = 0
    current_norm = None
    while True:
        prev_norm = current_norm
        for region in regions:
            for module_set in region:
                for module in module_set:
                    previous_name_to_module[module] = copy.deepcopy(name_to_module[module])
            _cross_layer_equalization([name_to_module[n] for n in region[0]], [name_to_module[n] for n in region[1]])
        current_norm = _norm(name_to_module, previous_name_to_module)
        iteration += 1
        if iteration > max_iterations:
            break
        if _norm is not None and prev_norm is not None:
            diff = torch.abs(prev_norm - current_norm)
            print(diff)
            if diff < threshold:
                break
    return model


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
        if node.op == 'call_module' and isinstance(get_module(graph_model, node.target), tuple(_supported_layers.keys())):
            if walk_forward:
                sinks.add(node.target)
            else:
                srcs.add(node.target)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
        elif node.op == 'call_module' and isinstance(get_module(graph_model, node.target), _scale_invariant_layers):
            if walk_forward:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
            else:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        elif (node.op == 'call_method' and node.target in _residual_methods
            or node.op == 'call_function' and node.target in _residual_fns):
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        else:
            continue


def _extract_regions(graph_model: GraphModule):
    regions = set()
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if isinstance(module, tuple(_supported_layers.keys())):
                srcs, sinks = {node.target}, set()
                walk_region(graph_model, node, set(), srcs, sinks, walk_forward=True)
                if sinks:
                    # each region should appear only once, so to make it hashable
                    # we convert srcs and sinks to ordered lists first, and then to tuples
                    regions.add((tuple(sorted(srcs)), tuple(sorted(sinks))))
    # for clarity, sort by the of the first source
    regions = sorted(regions, key=lambda region: region[0][0])
    return regions


def equalize_graph(graph_model: GraphModule, threshold=1e-4, max_iterations=100):
    regions = _extract_regions(graph_model)
    graph_model = _equalize(graph_model, regions, threshold, max_iterations)
    return graph_model