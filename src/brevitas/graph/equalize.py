# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple
from copy import deepcopy
from functools import partial
import operator
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.fx import Node
from brevitas.graph.utils import get_module

from .base import GraphTransform

__all__ = ['EqualizeGraph']

EPSILON = 1e-9

Region = namedtuple('Region', ['srcs', 'sinks'])

_supported_layers = (
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MultiheadAttention,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d)

_scale_invariant_layers = (
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.ReLU,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d)

_scale_invariant_op = (torch.mul, operator.mul, operator.imul, operator.__mul__, operator.__imul__)

_select_op = (operator.getitem, operator.__getitem__)

_residual_methods = ('add', 'add_')

_residual_fns = (torch.add, operator.add, operator.iadd, operator.__add__, operator.__iadd__)

_batch_norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

WeightBiasTuple = namedtuple('WeightBiasTuple', ['weight', 'bias'], defaults=[None])


def _select_scale_computation_fn(
        scale_computation_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if scale_computation_type == 'maxabs':
        return _channel_maxabs
    elif scale_computation_type == 'range':
        return _channel_range
    else:
        raise RuntimeError(f"Scale computation type {scale_computation_type} not supported")


def _channel_range(inp: torch.Tensor) -> torch.Tensor:
    mins, _ = inp.min(dim=1)
    maxs, _ = inp.max(dim=1)
    out = maxs - mins
    # correct corner case where where all weights along a channel have the same value
    # e.g. when a mean/nn.AvgPool/nn.AdaptiveAvgPool is converted to a depth-wise conv
    out = torch.where(out == 0., torch.mean(inp, dim=1), out)

    # convert to positive range, in case any of the values are negative,
    # highly likely in cases when there is only one value per channel, such as in Batch Norm
    out = torch.abs(out)
    return out


def _channel_maxabs(inp: torch.Tensor) -> torch.Tensor:
    out = torch.max(torch.abs(inp), dim=1)[0]
    return out


def _get_size(axes: Dict[nn.Module, int]) -> int:
    m0, axis0 = list(axes.items())[0]
    size = m0.weight.size(axis0)
    for m, axis in axes.items():
        if m.weight.size(axis) != size:
            return None
    return size


def _get_input_axis(module: nn.Module) -> Optional[int]:
    """
    Given a sink module, determine the axis associated to the input channels.
    Return None if not supported.
    """
    if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
        return 1
    elif isinstance(module, _batch_norm):
        return 0
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if module.groups == 1:
            return 1
        elif module.groups == module.out_channels:
            return 0
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if module.groups == 1:
            return 0
        elif module.groups == module.out_channels:
            return 1
    elif isinstance(module, nn.LayerNorm):
        # We assume normalization happens only along the channel dimension
        if len(module.weight.shape) == 1:
            return 0
        else:
            return None
    else:
        return None


def _get_output_axis(module: nn.Module) -> Optional[int]:
    """
    Given a source module, determine the axis associated to the output channels.
    Return None if not supported.
    """
    if isinstance(module,
                  (nn.Linear,
                   nn.Conv1d,
                   nn.Conv2d,
                   nn.Conv3d,
                   nn.MultiheadAttention,
                   nn.BatchNorm1d,
                   nn.BatchNorm2d,
                   nn.BatchNorm3d)):
        return 0
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return 1
    elif isinstance(module, nn.LayerNorm):
        # We assume normalization happens only along the channel dimension
        if len(module.weight.shape) == 1:
            return 0
        else:
            return None
    else:
        return None


def _combine_weights_bias(
        weight: torch.Tensor, bias_shrinkage: Union[float, str],
        bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Combine weights and bias before graph equalizattion
    This method merges the weight and bias of the sources, so that the resulting equalizer scale factor
    is influenced also by the magnitude of the bias, mitigated by a shrink factor.
    This technique avoids that, after the equalization procedure, the bias values become too big,
    negatively impacting the quantization accuracy.
    The bias shrinkage factor regulates how much the bias magnitude will affect the subsequence
    calculation of the scale.
    """
    if bias is None:
        return weight.data

    bias = bias.data
    weight = deepcopy(weight).view(weight.shape[0], -1)
    bias = deepcopy(bias).view(-1, 1)

    weight = torch.where(torch.abs(weight) < EPSILON, torch.tensor(EPSILON).type_as(weight), weight)
    factor = torch.abs(bias) / torch.abs(weight)

    # From https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/nndct_shared/optimization/commander.py#L450
    if bias_shrinkage == 'vaiq':
        if torch.abs(bias).max() < 10 and (torch.abs(bias).max() / torch.abs(weight).max()) < 20:
            if torch.median(factor) > 100 or torch.mean(factor) > 1000:
                shrink_factor = 5.
            else:
                shrink_factor = 2.
        else:
            if torch.median(factor) > 30 or torch.mean(factor) > 500:
                shrink_factor = 20.
            elif torch.median(factor) > 15 or torch.mean(factor) > 250:
                shrink_factor = 10.
            else:
                shrink_factor = 5.
    elif isinstance(bias_shrinkage, (int, float)):
        shrink_factor = bias_shrinkage
    else:
        raise RuntimeError(f"{bias_shrinkage} not supported.")
    weight_bias = torch.cat([weight, bias / shrink_factor], 1)
    return weight_bias


def _cross_layer_equalization(
        srcs: List[nn.Module],
        sinks: List[nn.Module],
        merge_bias: bool,
        bias_shrinkage: Union[float, str],
        scale_computation_type: str) -> torch.Tensor:
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    # Determine device and type of tensors
    device = next(srcs[0].parameters()).device
    dtype = next(srcs[0].parameters()).dtype

    src_axes = {}
    sink_axes = {}

    for i, module in enumerate(srcs):
        # If module is not supported, do not perform graph equalization
        if not isinstance(module, _supported_layers):
            return torch.tensor(1., dtype=dtype, device=device)
        if isinstance(module, nn.MultiheadAttention):
            srcs[i] = module.out_proj
        src_axes[srcs[i]] = _get_output_axis(module)

    for i, module in enumerate(sinks):
        # If module is not supported, do not perform graph equalization
        if not isinstance(module, _supported_layers):
            return torch.tensor(1., dtype=dtype, device=device)
        # For MultiheadAttention, we support only self-attetion
        if isinstance(module, nn.MultiheadAttention) and hasattr(sinks[i], 'in_proj_weight'):
            # For sinks, we only need to modify the weight but not the bias
            sinks[i] = WeightBiasTuple(module.in_proj_weight)
        elif isinstance(module, nn.MultiheadAttention) and not hasattr(sinks[i], 'in_proj_weight'):
            return torch.tensor(1., dtype=dtype, device=device)
        sink_axes[sinks[i]] = _get_input_axis(module)

    # Check if any of the axis is None, which means that the module is not supported.
    # In that case, do not perform graph equalization
    if None in [*src_axes.values(), *sink_axes.values()]:
        return torch.tensor(1., dtype=dtype, device=device)

    src_size = _get_size(src_axes)
    sink_size = _get_size(sink_axes)

    # Check if any of the src_size or sink_size is None, which means that the some of the
    # sources or sinks do not have the same size as the others.
    # Similarly, exit if source and sink have different different sizes
    if None in [src_size, sink_size] or src_size != sink_size:
        return torch.tensor(1., dtype=dtype, device=device)
    transpose = lambda module, axis: module.weight if axis == 0 else module.weight.transpose(0, 1)
    scale_fn = _select_scale_computation_fn(scale_computation_type)
    if merge_bias:
        src_weights = [
            _combine_weights_bias(transpose(m, axis), bias_shrinkage, m.bias) for m,
            axis in src_axes.items()]
    else:
        src_weights = [transpose(m, axis) for m, axis in src_axes.items()]
    sink_weights = [transpose(m, axis) for m, axis in sink_axes.items()]
    srcs_range = scale_fn(torch.cat([w.reshape(w.size(0), -1) for w in src_weights], 1))
    sinks_range = scale_fn(torch.cat([w.reshape(w.size(0), -1) for w in sink_weights], 1))
    sinks_range += EPSILON

    scaling_factors = torch.sqrt(srcs_range / sinks_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    for module, axis in src_axes.items():
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data * inverse_scaling_factors.view_as(module.bias)
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        module.weight.data = module.weight.data * torch.reshape(
            inverse_scaling_factors, src_broadcast_size)
    for module, axis in sink_axes.items():
        src_broadcast_size = [1] * module.weight.ndim
        src_broadcast_size[axis] = module.weight.size(axis)
        if isinstance(module, _batch_norm):
            # We re-compute the bias as function of running_mean and running_var to adjust the
            # additive factor for equalization.
            additive_factor = module.running_mean.data * module.weight.data / torch.sqrt(
                module.running_var.data + module.eps)
            module.bias.data = module.bias.data + additive_factor * (scaling_factors - 1)
        module.weight.data = module.weight.data * torch.reshape(scaling_factors, src_broadcast_size)

    return scaling_factors


def _equalize(
        model: GraphModule,
        regions: Set[Tuple[str]],
        iterations: int,
        threshold: float,
        merge_bias: bool,
        bias_shrinkage: Union[str, float],
        scale_computation_type: str) -> GraphModule:
    """
    Generalized version of section 4.1 of https://arxiv.org/pdf/1906.04721.pdf
    """
    name_to_module: Dict[str, nn.Module] = {}
    name_set = {name for region in regions for module_set in region for name in module_set}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    for i in range(iterations):
        scale_factor_max = None
        for region in regions:
            scale_factors_region = _cross_layer_equalization(
                [name_to_module[n] for n in region.srcs], [name_to_module[n] for n in region.sinks],
                merge_bias,
                bias_shrinkage,
                scale_computation_type)
            scale_factor_region_max = torch.max(torch.abs(1 - scale_factors_region))
            if scale_factor_max is not None:
                scale_factor_max = torch.max(scale_factor_max, scale_factor_region_max)
            else:
                scale_factor_max = scale_factor_region_max
        if threshold is not None and scale_factor_max < threshold:
            break
    return model


def _is_supported_module(graph_model: GraphModule, node: Node) -> bool:
    if node.op == 'call_module':
        module = get_module(graph_model, node.target)
        if isinstance(module, _supported_layers):
            # We support only self-attention
            if isinstance(module, nn.MultiheadAttention):
                return all([node.all_input_nodes[0].name == n.name for n in node.all_input_nodes])
            return True
    return False


def _is_scale_invariant_module(graph_model: GraphModule, node: Node) -> bool:
    return node.op == 'call_module' and isinstance(
        get_module(graph_model, node.target), _scale_invariant_layers)


def _is_scale_invariant_function(node: Node) -> bool:
    return node.op == 'call_function' and node.target in _scale_invariant_op + _select_op


def _is_reshaping_op(node: Node) -> bool:
    return (
        node.op == 'call_function' and node.target in [torch.flatten, torch.reshape] or
        node.op == 'call_method' and node.target in ['view', 'reshape', 'flatten'])


def walk_region(
        graph_model: GraphModule,
        starting_node: Node,
        history: Set[Node],
        srcs: Set[str],
        sinks: Set[str],
        walk_forward: bool):
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
                module = get_module(graph_model, node.target)
                # It is not possible to equalize through LayerNorm as sink
                if not isinstance(module, nn.LayerNorm):
                    sinks.add(node.target)
            else:
                srcs.add(node.target)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
        elif _is_scale_invariant_module(graph_model, node) or _is_scale_invariant_function(node):
            if walk_forward:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
            else:
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
                walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        elif (node.op == 'call_method' and node.target in _residual_methods or
              node.op == 'call_function' and node.target in _residual_fns):
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=True)
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=False)
        elif _is_reshaping_op(node):
            walk_region(graph_model, node, history, srcs, sinks, walk_forward=walk_forward)
        else:
            continue


def _extract_regions(graph_model: GraphModule) -> Set[Tuple[str]]:
    regions = set()
    for node in graph_model.graph.nodes:
        if _is_supported_module(graph_model, node):
            srcs, sinks = {node.target}, set()
            walk_region(graph_model, node, set(), srcs, sinks, walk_forward=True)
            if sinks:
                # each region should appear only once, so to make it hashable
                # we convert srcs and sinks to ordered lists first, and then to tuples
                regions.add(Region(tuple(sorted(srcs)), tuple(sorted(sinks))))
    return regions


class EqualizeGraph(GraphTransform):

    def __init__(
            self,
            iterations: int = 10,
            threshold: float = 0.05,
            return_regions: bool = False,
            merge_bias: bool = True,
            bias_shrinkage: Union[float, str] = 'vaiq',
            scale_computation_type: str = 'maxabs') -> None:
        super(EqualizeGraph, self).__init__()
        self.iterations = iterations
        self.return_regions = return_regions
        self.merge_bias = merge_bias
        self.bias_shrinkage = bias_shrinkage
        self.threshold = threshold
        self.scale_computation_type = scale_computation_type

    def apply(self,
              graph_model: GraphModule) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(graph_model)
        if len(regions) > 0:
            graph_model = _equalize(
                graph_model,
                regions,
                self.iterations,
                self.threshold,
                self.merge_bias,
                self.bias_shrinkage,
                self.scale_computation_type)
        if self.return_regions:
            return graph_model, regions
        else:
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
            module.bias = nn.Parameter(tensor)

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
            if (_is_supported_module(graph_model, node) and node.next.op == 'call_module' and
                    isinstance(get_module(graph_model, node.next.target), _batch_norm)):
                node_next = node.next.next
                while _is_scale_invariant_module(graph_model,
                                                 node_next) or _is_reshaping_op(node_next):
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
