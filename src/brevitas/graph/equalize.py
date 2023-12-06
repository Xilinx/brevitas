# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

<<<<<<< HEAD
from abc import ABC
from abc import abstractmethod
=======
from collections import namedtuple
from collections import OrderedDict
from copy import deepcopy
>>>>>>> Feat (graph/equalize): extended graph equalization
from dataclasses import dataclass
from dataclasses import field
from functools import partial
import operator
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

import torch
from torch.fx import GraphModule as TorchGraphModule
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.fx import Node
from brevitas.graph.base import GraphTransform
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.utils import get_module
from brevitas.graph.utils import get_node
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.quant_scale_bias import ScaleBias
from brevitas.utils.torch_utils import KwargsForwardHook

from .base import GraphTransform
from .base import InsertModuleCallAfter

__all__ = ['GraphActivationEqualization', 'LayerwiseActivationEqualization', 'EqualizeGraph']

EPSILON = 1e-9

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
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.Identity,
    nn.ReLU,
    nn.LeakyReLU)

_scale_invariant_op = (
    torch.mul,
    operator.mul,
    operator.imul,
    operator.__mul__,
    operator.__imul__,
    torch.nn.functional.interpolate)

_select_op = (operator.getitem, operator.__getitem__)

_reshaping_op = ('view', 'reshape', 'flatten', 'contiguous', torch.reshape, torch.flatten)

_scale_varying_activations = (
    torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU6, torch.nn.GELU, torch.nn.SiLU)

_residual_methods = ('add', 'add_')

_residual_fns = (torch.add, operator.add, operator.iadd, operator.__add__, operator.__iadd__)

_batch_norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

_ignore_ops = (getattr, 'size')


# Required for being hashable
@dataclass(eq=True, frozen=True)
class WeightBiasTuple:
    weight: nn.Module = None
    bias: nn.Module = None


# Required for being hashable
@dataclass(eq=True, frozen=True)
class Region:
    srcs: Dict = field(default_factory=dict)
    sinks: Dict = field(default_factory=dict)
    acts: Tuple = field(default_factory=tuple)


@dataclass
class WalkRegionState:
    srcs: Dict = field(default_factory=dict)
    sinks: Dict = field(default_factory=dict)
    acts: Set = field(default_factory=set)
    history: set = field(default_factory=set)
    add_mul_node: bool = False
    offset: int = 0
    update_offset: bool = False


# Start and End identify the starting and ending channels of the weight matrix that need to be
# equalized.
# Offset refers to the relative position of these channels with respect to
# the other matrices' channels that are equalized simultaneously.
# Source matrix are always fully equalized, while sinks can be partially equalized.
@dataclass
class EqualizationIndexes:
    start: int = 0
    end: int = 0
    offset: int = 0


def __str__(self):
    return str(self.start) + '_' + str(self.end) + '_' + str(self.offset)


_UNSUPPORTED_OP = object()


def _select_scale_computation_fn(
        scale_computation_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if scale_computation_type == 'maxabs':
        return _channel_maxabs
    elif scale_computation_type == 'range':
        return _channel_range
    else:
        raise RuntimeError(f"Scale computation type {scale_computation_type} not supported")


class activation_equalization_mode:

    def __init__(self, model, alpha, add_mul_node=True, layerwise=True, enabled=True) -> None:
        self.model = model
        self.alpha = alpha
        self.enabled = enabled
        self.add_mul_node = add_mul_node
        if layerwise:
            if not self.add_mul_node:
                raise ValueError("Layerwise activation equalization requires add_mul_node")
            self.graph_act_eq = LayerwiseActivationEqualization(self.model)
        else:
            if not isinstance(self.model, (TorchGraphModule, GraphModule)):
                raise TypeError(
                    "A Torch FX representation of the model is needed for Graph Activation Equalization"
                )
            self.graph_act_eq = GraphActivationEqualization(self.model, self.add_mul_node)
        self.scale_factors = None

    def __enter__(self):
        if self.enabled:
            self.graph_act_eq.setup()
        return self

    def __exit__(self, type, value, traceback):
        if self.enabled:
            self.scale_factors = self.graph_act_eq.apply(self.alpha)
        return True  # To propagate exceptions


def dict_name_to_module(model, regions):
    name_to_module: Dict[str, torch.nn.Module] = {}

    name_set = set()
    for region in regions:
        for name in region.srcs:
            name_set.add(name)
        for name in region.sinks:
            name_set.add(name)
        for name in region.acts:
            name_set.add(name)
    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    return name_to_module


def _channel_range(inp: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mins, _ = inp.min(dim=dim)
    maxs, _ = inp.max(dim=dim)
    out = maxs - mins
    # correct corner case where where all weights along a channel have the same value
    # e.g. when a mean/nn.AvgPool/nn.AdaptiveAvgPool is converted to a depth-wise conv
    out = torch.where(out == 0., torch.mean(inp, dim=dim), out)

    # convert to positive range, in case any of the values are negative,
    # highly likely in cases when there is only one value per channel, such as in Batch Norm
    out = torch.abs(out)
    return out


def _channel_maxabs(inp: torch.Tensor, dim: int = 1) -> torch.Tensor:
    out = torch.max(torch.abs(inp), dim=dim)[0]
    return out


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


def _get_act_axis(sink_module: nn.Module) -> Optional[int]:
    """
    Given a layer, return activation axis associated to the feature dim.
    Even though BatchNorm layers are supported, it gives no information about the feature dim.
    """
    if isinstance(sink_module, (nn.Linear, nn.MultiheadAttention)):
        return -1
    elif isinstance(sink_module,
                    (nn.Conv1d,
                     nn.Conv2d,
                     nn.Conv3d,
                     nn.ConvTranspose1d,
                     nn.ConvTranspose2d,
                     nn.ConvTranspose3d)):
        return 0
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
    weight = weight.reshape(weight.shape[0], -1)
    bias = bias.reshape(-1, 1)

    weight = torch.where(
        torch.abs(weight) <= EPSILON, torch.tensor(EPSILON).type_as(weight), weight)
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


def transpose(module: torch.nn.Module, axis: int):
    """
    Given a module and an axis, this function re-arranges the module's weights so that the axis and
    the first dimension are swapped.
    """
    shape = list(range(module.weight.ndim))
    axis = shape[axis]
    shape.insert(0, axis)
    del shape[axis + 1]
    return module.weight.permute(shape)


def _cross_layer_equalization(
        srcs: Dict[nn.Module, List[int]],
        sinks: Dict[nn.Module, List[int]],
        merge_bias: bool,
        scale_computation_type: str,
        bias_shrinkage: Optional[Union[float, str]] = None,
        list_of_act_val: Optional[torch.Tensor] = None,
        list_of_insert_mul_node_fn: Optional[List[Callable]] = None,
        alpha: float = 0.5) -> torch.Tensor:
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """

    # Determine device and type of tensors
    device = next(sinks['sinks0'][0].parameters()).device
    dtype = next(sinks['sinks0'][0].parameters()).dtype

    # If equalization criteria are not met, we return a scalar one to indicate that no equalization
    # has been performed
    def _no_equalize():
        print("No Eq")
        return torch.tensor(1., dtype=dtype, device=device)

    act_sink_axes = {}
    act_sources_axes = {}
    single_module = list(sinks.values())[0][0]
    device = next(single_module.parameters()).device
    dtype = next(single_module.parameters()).dtype

    max_shape_srcs = 0
    for name, (k, v) in srcs.items():
        max_shape_srcs = max(max_shape_srcs, v.end + v.offset)
    max_shape_sinks = 0
    for name, (k, v) in sinks.items():
        max_shape_sinks = max(max_shape_sinks, v.offset + (v.end - v.start))
    # Exit if source and sink have different sizes

    if max_shape_srcs != max_shape_sinks and len(srcs) > 0:
        return _no_equalize()

    # for i, module in enumerate(srcs):
    src_axes = {}
    for i, (name, (module, indexes)) in enumerate(srcs.items()):
        # If module is not supported, do not perform graph equalization
        axis = _get_output_axis(module)
        act_sources_axes[name] = _get_act_axis(module)
        if not isinstance(module, _supported_layers):
            return _no_equalize()
        if isinstance(module, nn.MultiheadAttention):
            module = module.out_proj
            srcs[name] = (module, indexes)
        src_axes[name] = (module, axis)

    sink_axes = {}
    for i, (name, (module, indexes)) in enumerate(sinks.items()):
        axis = _get_input_axis(module)
        act_sink_axes[name] = _get_act_axis(module)
        # If module is not supported, do not perform graph equalization
        if not isinstance(module, _supported_layers):
            return _no_equalize()
        # For MultiheadAttention, we support only self-attetion
        if isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
            # For sinks, we only need to modify the weight but not the bias
            module = WeightBiasTuple(module.in_proj_weight)
            sinks[name] = (module, indexes)
        elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is None:
            return _no_equalize()
        sink_axes[name] = (module, axis)
    # If act_val is enabled, use source or sink weights to determine the activation channel
    # For example, if the source is BatchNorm, we need to use the information coming from the sinks
    if list_of_act_val is not None:
        list_of_sink_axes = [x for x in list(act_sink_axes.values()) if x is not None]
        list_of_source_axes = [x for x in list(act_sources_axes.values()) if x is not None]
        if len(list_of_sink_axes) > 0:
            act_axis = list_of_sink_axes[0]
        elif len(list_of_source_axes) > 0:
            act_axis = list_of_source_axes[0]
        else:
            return _no_equalize()
        # If there is a mismatch in the activation channel (e.g. a transpose/flatten op in between),
        # do not perform equalization
        if any([act_axis != axis for axis in list_of_source_axes + list_of_sink_axes]):
            return _no_equalize()

    # Check if any of the axis is None, which means that the module is not supported.
    # In that case, do not perform graph equalization
    axes_to_check = [*src_axes.values(), *sink_axes.values()]
    if None in axes_to_check:
        return _no_equalize()

    scale_fn = _select_scale_computation_fn(scale_computation_type)
    sink_weights = {name: transpose(m, axis) for name, (m, axis) in sink_axes.items()}
    srcs_range = -1 * torch.ones(max_shape_srcs, device=device, dtype=dtype)
    sinks_range = -1 * torch.ones(max_shape_sinks, device=device, dtype=dtype)
    for k, v in sink_weights.items():
        # Sinks can be partially equalized, thus we need to select
        # only the channels we are interested in
        indexes = sinks[k][1]
        # Compute the range of the channels we need to equalize
        weight_range = scale_fn(v.reshape(v.size(0), -1))[indexes.start:indexes.end]
        # Compute the numbers of channels we are equalizing
        channel_range = indexes.end - indexes.start
        # Use the offset and the range to update the correct range in the sinks
        sinks_range[indexes.offset:indexes.offset + channel_range] = torch.max(
            sinks_range[indexes.offset:indexes.offset + channel_range], weight_range)
    sinks_range = torch.clamp(sinks_range, EPSILON)

    # Determine the srcs_range based on where we are performing activation equalization or
    # weight equalization
    if list_of_act_val is not None:
        list_of_act_val_shapes = [act_val.shape for act_val in list_of_act_val]
        if len(list_of_act_val_shapes) > 0:
            shape_0 = list_of_act_val_shapes[0]
            if any(shape_0 != shape for shape in list_of_act_val_shapes):
                return _no_equalize()
        list_of_act_val = [
            transpose(WeightBiasTuple(act_val), act_axis) for act_val in list_of_act_val]
        srcs_range = scale_fn(
            torch.cat([act_val.reshape(act_val.size(0), -1) for act_val in list_of_act_val], 1))
    else:
        if merge_bias:
            src_weights = {
                name: _combine_weights_bias(transpose(m, axis), bias_shrinkage, m.bias)
                for name, (m, axis) in src_axes.items()}
        else:
            src_weights = {name: transpose(m, axis) for name, (m, axis) in src_axes.items()}
        for k, v in src_weights.items():
            # Srcs are always fully equalized, thus we simply need to apply the offset to position them
            # correctly with respect to the other srcs matrices.
            indexes = srcs[k][1]
            channel_start = indexes.offset + indexes.start
            channel_end = indexes.offset + indexes.end
            weight_range = scale_fn(v.reshape(v.size(0), -1))
            srcs_range[channel_start:channel_end] = torch.max(
                srcs_range[channel_start:channel_end], weight_range)

    # If there is a mismatch between srcs and sinks values, exit
    if srcs_range.shape != sinks_range.shape:
        warnings.warn(
            "Detected source and sink with non compatible shapes, equalization is skipped")
        return _no_equalize()

    # Instead of clipping very low values, which would cause their reciprocal to be very large
    # thus hindering quantization, we set both sources and sinks to one,
    # which is the no-op equivalent for equalization.
    channelwise_no_equalize = (sinks_range <= EPSILON) | (srcs_range <= EPSILON)
    sinks_range = torch.where(
        channelwise_no_equalize, torch.tensor(1., dtype=dtype, device=device), sinks_range)
    srcs_range = torch.where(
        channelwise_no_equalize, torch.tensor(1., dtype=dtype, device=device), srcs_range)

    srcs_range = torch.pow(srcs_range, alpha)
    sinks_range = torch.pow(sinks_range, 1 - alpha)
    scaling_factors = srcs_range / sinks_range
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    if list_of_act_val is not None and list_of_insert_mul_node_fn is not None:
        for act_val_shape, insert_mul_node_fn in zip(list_of_act_val_shapes, list_of_insert_mul_node_fn):
            insert_mul_node_fn(inverse_scaling_factors, act_val_shape, act_axis)
    if len(src_axes) > 0:
        for name, (module, axis) in src_axes.items():
            indexes = srcs[name][1]
            channel_start = indexes.offset + indexes.start
            channel_end = indexes.offset + indexes.end
            if hasattr(module, 'bias') and module.bias is not None:
                _update_weights(
                    module,
                    module.bias.clone() *
                    inverse_scaling_factors[channel_start:channel_end].view_as(module.bias),
                    attr='bias')
            src_broadcast_size = [1] * module.weight.ndim
            src_broadcast_size[axis] = module.weight.size(axis)

            _update_weights(
                module,
                module.weight.clone() * torch.reshape(
                    inverse_scaling_factors[channel_start:channel_end], src_broadcast_size),
                attr='weight')
    for module, axis in sink_axes.items():
        sink_broadcast_size = [1] * module.weight.ndim
        sink_broadcast_size[axis] = module.weight.size(axis)
        indexes = sinks[name][1]
        channel_range = indexes.end - indexes.start
        partial_scaling = torch.ones(module.weight.size(axis), device=device, dtype=dtype)
        # We replace the scaling factors of the channels we need to equalize, leaving the other to
        # one (i.e., no equalization)
        partial_scaling[indexes.start:indexes.end] = scaling_factors[indexes.offset:indexes.offset +
                                                                     channel_range]
        if isinstance(module, _batch_norm):
            # We re-compute the bias as function of running_mean and running_var to adjust the
            # additive factor for equalization.
            additive_factor = module.running_mean.data * module.weight.data / torch.sqrt(
                module.running_var.data + module.eps)
            _update_weights(
                module, module.bias.clone() + additive_factor * (partial_scaling - 1), attr='bias')
        _update_weights(
            module,
            module.weight.clone() * torch.reshape(partial_scaling, sink_broadcast_size),
            attr='weight')

    return scaling_factors


def _update_weights(original_module, new_value, attr='weight'):
    if isinstance(original_module, WeightBiasTuple):
        setattr(getattr(original_module, attr), 'data', new_value)
    else:
        setattr(original_module, attr, nn.Parameter(new_value))


def _organize_region(region, name_to_module, type):
    region_dict = {}
    region = getattr(region, type)
    for i, (k, v) in enumerate(region.items()):
        name = type + str(i)
        k = k.split('$')[0]
        region_dict[name] = (name_to_module[k], v)
    return region_dict


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
    name_set = set()
    for region in regions:
        for name in region.srcs.keys():
            name_set.add(name.split("$")[0])
        for name in region.sinks.keys():
            name_set.add(name.split("$")[0])

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    for i in range(iterations):
        scale_factor_max = None
        for ii, region in enumerate(regions):
            srcs_dict = _organize_region(region, name_to_module, 'srcs')
            sinks_dict = _organize_region(region, name_to_module, 'sinks')
            scale_factors_region = _cross_layer_equalization(
                srcs_dict,
                sinks_dict,
                merge_bias=merge_bias,
                bias_shrinkage=bias_shrinkage,
                scale_computation_type=scale_computation_type)
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
                kwargs = dict(node.kwargs)
                kwargs.update(zip(module.forward.__code__.co_varnames[1:], node.args))
                return kwargs['query'].name == kwargs['key'].name == kwargs['value'].name
            return True
    return False


def _is_scale_invariant_module(graph_model: GraphModule, node: Node) -> bool:
    return node.op == 'call_module' and isinstance(
        get_module(graph_model, node.target), _scale_invariant_layers)


def _is_scale_varying_activation(graph_model, node):
    return node.op == 'call_module' and isinstance(
        get_module(graph_model, node.target), _scale_varying_activations)


def _is_scale_invariant_function(node: Node) -> bool:
    out = node.op == 'call_function' and node.target in _scale_invariant_op + _select_op
    if node.target == torch.nn.functional.interpolate:
        out &= node.kwargs.get('mode', None) == 'nearest'
    return out


def _is_reshaping_op(node: Node) -> bool:
    return node.target in _reshaping_op


def get_weight_source(module_list):
    transpose = lambda module, axis: module.weight if axis == 0 else module.weight.transpose(0, 1)
    for i, module in enumerate(module_list):
        if isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'out_proj'):
                module_list[i] = module.out_proj
            else:
                raise RuntimeError("Configuration for Multiheadattention not supported")
    srcs_axes = {module: _get_output_axis(module) for module in module_list}
    weight = [transpose(m, axis) for m, axis in srcs_axes.items()]
    return weight


def get_weight_sink(module_list):
    transpose = lambda module, axis: module.weight if axis == 0 else module.weight.transpose(0, 1)
    for i, module in enumerate(module_list):
        if isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight'):
                module_list[i] = WeightBiasTuple(module.in_proj_weight)
            else:
                raise RuntimeError("Configuration for Multiheadattention not supported")
    sinks_axes = {module: _get_input_axis(module) for module in module_list}
    weight = [transpose(m, axis) for m, axis in sinks_axes.items()]
    return weight


def find_srcs_channel_dim(model, inp_node):
    if _is_supported_module(model, inp_node):
        # If we meet a supported module, determine the channel shape
        module = get_module(model, inp_node.target)
        # Since we are walking up, we consider the module as srcs
        weight = get_weight_source([module])
        channel = weight[0].shape[0]
        return channel
    elif _is_add(inp_node):
        # If it's add, we need the channel shape of one of the branches, since they are all the same
        return find_srcs_channel_dim(model, inp_node.all_input_nodes[0])
    elif _is_cat(inp_node):
        total_channels = 0
        # If it's cat, we need to sum the channel shape of all the branches
        for n in inp_node.all_input_nodes:
            total_channels += find_srcs_channel_dim(model, n)
        return total_channels
    elif _is_scale_invariant_module(model, inp_node) or _is_scale_invariant_function(inp_node):
        return find_srcs_channel_dim(model, inp_node.all_input_nodes[0])
    else:
        return _UNSUPPORTED_OP


def cat_handler(graph_model: GraphModule, starting_node: Node, state: WalkRegionState):

    state.srcs.clear()
    state.sinks.clear()
    state.history.clear()
    state.srcs[starting_node.target] = _UNSUPPORTED_OP
    state.update_offset = True
    state.offset = 0
    find_srcs(graph_model, starting_node, state)
    state.update_offset = False
    state.offset = 0
    find_sinks(graph_model, starting_node, state)


def _is_cat(node):
    return node.target in (torch.cat,)


def _is_cat_in_srcs(srcs):
    out = False
    for src in srcs:
        out = out or src in (torch.cat,)
    return out


def _is_add(node):
    return (
        node.op == 'call_method' and node.target in _residual_methods or
        node.op == 'call_function' and node.target in _residual_fns)


def find_srcs(graph_model: GraphModule, starting_node: Node,
              state: WalkRegionState) -> Dict[str, Set]:
    node_list = starting_node.all_input_nodes
    update_offset_state = state.update_offset
    for node in node_list:
        # we keep a history of how the graph has been walked already, invariant to the direction,
        # to avoid getting stuck in a loop
        path = (node, starting_node)
        module = None
        if path not in state.history:
            state.history.add(path)
        else:
            continue
        if _is_supported_module(graph_model, node):
            module = get_module(graph_model, node.target)
            weight = get_weight_source([module])
            eq_indexes = EqualizationIndexes(0, weight[0].shape[0], state.offset)
            # state.srcs.add((node.target, eq_indexes))
            full_source_name = node.target + '$' + str(eq_indexes)
            state.srcs[full_source_name] = eq_indexes
            # After we found a source, we need to check if it branches into multiple sinks
            find_sinks(graph_model, node, state)
            state.offset = state.offset if not state.update_offset else state.offset + weight[
                0].shape[0]
        elif _is_scale_invariant_module(
                graph_model, node) or _is_scale_invariant_function(node) or _is_reshaping_op(node):
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
        elif (node.op == 'call_method' and node.target in _residual_methods or
              node.op == 'call_function' and node.target in _residual_fns):
            state.update_offset = False
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
            state.update_offset = update_offset_state
        elif _is_cat(node):
            # We have never encoutered cat
            if not _is_cat_in_srcs(state.srcs):
                # We restart the region search starting from the cat
                cat_handler(graph_model, node, state)
            # We have encoutered cat already
            else:
                state.update_offset = False
                find_sinks(graph_model, node, state)
                state.update_offset = True
                find_srcs(graph_model, node, state)
                state.update_offset = update_offset_state
        elif node.target in _ignore_ops:
            continue
        else:
            # If we meet an unrecognized op, we add None to invalidate the region
            state.sinks[_UNSUPPORTED_OP] = _UNSUPPORTED_OP


def find_sinks(graph_model: GraphModule, starting_node: Node,
               state: WalkRegionState) -> Dict[str, Set]:
    node_list = starting_node.users
    update_offset_state = state.update_offset
    for node in node_list:
        # we keep a history of how the graph has been walked already, invariant to the direction,
        # to avoid getting stuck in a loop
        # Note that the path is inverted with respect to find_srcs
        path = (starting_node, node)
        if path not in state.history:
            state.history.add(path)
        else:
            continue
        if _is_supported_module(graph_model, node):
            module = get_module(graph_model, node.target)
            weight = get_weight_sink([module])
            eq_indexes = EqualizationIndexes(0, weight[0].shape[0], state.offset)
            # It is not possible to equalize through LayerNorm as sink
            if isinstance(module, (nn.LayerNorm,)):
                # state.sinks.add((_UNSUPPORTED_OP, _UNSUPPORTED_OP))
                state.sinks[_UNSUPPORTED_OP] = _UNSUPPORTED_OP
            else:
                full_sink_name = node.target + '$' + str(eq_indexes)
                state.sinks[full_sink_name] = eq_indexes
                # state.sinks[node.target] = eq_indexes
        elif _is_scale_invariant_module(
                graph_model, node) or _is_scale_invariant_function(node) or _is_reshaping_op(node):
            find_sinks(graph_model, node, state)
        elif (node.op == 'call_method' and node.target in _residual_methods or
              node.op == 'call_function' and node.target in _residual_fns):
            state.update_offset = False
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
            state.update_offset = update_offset_state
        elif _is_cat(node):
            # We have never encoutered cat
            if not _is_cat_in_srcs(state.srcs):
                # We restart the region search starting from the cat
                cat_handler(graph_model, node, state)
            # We have encoutered cat already
            else:
                # In this case we define all our sinks, and isolate only the channels we want
                # to equalize (start, end).
                # Furthermore, we need to consider the offset given by the sources of the second cat
                index = node.all_input_nodes.index(starting_node)
                channels = []
                for n in node.all_input_nodes:
                    channels.append(find_srcs_channel_dim(graph_model, n))
                start = sum(channels[:index])
                end = start + channels[index]
                new_state = WalkRegionState(offset=state.offset)
                find_sinks(graph_model, node, new_state)

                for k in new_state.sinks.keys():
                    state.sinks[k] = EqualizationIndexes(start, end, new_state.offset)
                state.srcs.update(new_state.srcs)
        elif node.target in _ignore_ops:
            continue
        else:
            # If we meet an unrecognized op, we add None to invalidate the region
            state.sinks[_UNSUPPORTED_OP] = _UNSUPPORTED_OP


def _extract_regions(
        graph_model: GraphModule,
        add_mul_node: bool = False,
        return_acts: bool = False) -> List[Region]:
    regions = list()
    regions_name = set()
    for node in graph_model.graph.nodes:
        if _is_supported_module(graph_model,
                                node) or (add_mul_node and
                                          _is_scale_varying_activation(graph_model, node)):
            state = WalkRegionState(add_mul_node=add_mul_node)
            if _is_scale_varying_activation(graph_model, node):
                state.acts.add(node.target)
            else:
                module = get_module(graph_model, node.target)
                weight = get_weight_source([module])
                eq_indexes = EqualizationIndexes(0, weight[0].shape[0], 0)
                full_source_name = node.target + '$' + str(eq_indexes)
                state.srcs[full_source_name] = eq_indexes
            find_sinks(graph_model, node, state)
            if state.sinks and _UNSUPPORTED_OP not in state.sinks.keys(
            ) and _UNSUPPORTED_OP not in state.srcs.keys():
                # Drop cat from the srcs
                state.srcs = {k: v for k, v in state.srcs.items() if k is not torch.cat}
                sorted_srcs = dict(sorted(state.srcs.items()))
                sorted_sinks = dict(sorted(state.sinks.items()))
                sorted_acts = tuple(sorted(state.acts))
                if return_acts:
                    region = Region(srcs=sorted_srcs, sinks=sorted_sinks, acts=sorted_acts)
                else:
                    region = Region(srcs=sorted_srcs, sinks=sorted_sinks)

                if region not in regions:
                    regions.append(region)
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


class ActivationEqualization(GraphTransform, ABC):

    def __init__(
            self, model: Union[nn.Module, GraphModule], scale_computation_type: str = 'maxabs'):
        self.model = model
        self.scale_computation_type = scale_computation_type

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def insert_mul_node(self):
        pass

    def create_mul_node(self, scale, shape, axis, batch_dim=0):
        broadcastable_shape = [1] * len(shape)
        broadcastable_shape[axis] = shape[axis]
        # Add Batch Dim
        broadcastable_shape.insert(batch_dim, 1)
        mul_factor = ScaleBias(
            num_features=shape[axis], bias=False, runtime_shape=broadcastable_shape)
        mul_factor.weight.data = scale
        return mul_factor

    def forward_stats_hook(self, module, *args, name, batch_dim=0, use_inp=True, **kwargs):
        # Check for MHA Cross attention, and if found, skip it
        kwargs.update(zip(module.forward.__code__.co_varnames[1:], args[:-1]))
        if 'query' in kwargs and 'key' in kwargs and 'value' in kwargs:
            if kwargs['query'].data_ptr() != kwargs['key'].data_ptr() != kwargs['value'].data_ptr():
                self.float_act_map[name] = None
                return

        possible_input_kwargs = ['input', 'inp', 'query']
        input_kwarg = [x for x in kwargs.keys() if x in possible_input_kwargs][0]
        if use_inp:
            x = kwargs[input_kwarg]
        elif not use_inp:
            x = args[-1]

        # Extra check for batch_dim
        if hasattr(x, 'names') and 'N' in x.names:
            batch_dim = x.names.index('N')

        self.batch_dim_act_map[name] = batch_dim

        input_scales = self.scale_fn(x, dim=batch_dim)
        if name not in self.float_act_map:
            self.float_act_map[name] = input_scales
        else:
            self.float_act_map[name] = torch.max(self.float_act_map[name], input_scales)

    def remove_hooks(self):
        for hook in self.hooks:
            ModuleInstanceToModuleInstance(hook, hook.module).apply(self.model)


class LayerwiseActivationEqualization(ActivationEqualization):

    def __init__(self, model, scale_computation_type: str = 'maxabs'):
        super(LayerwiseActivationEqualization, self).__init__(model, scale_computation_type)
        self.float_act_map = {}
        self.batch_dim_act_map = {}
        self.hooks = []
        self.add_mul_node = True

        regions = []
        self.find_module(model, regions)
        self.regions = regions

        if self.scale_computation_type == 'maxabs':
            self.scale_fn = _channel_maxabs
        elif self.scale_computation_type == 'range':
            self.scale_fn = _channel_range

    def find_module(self, model, regions: List):
        """
        Iterate through the model looking at immediate children of every module to look for supported modules.
        This allows us to stop the search when we meet a top-level module that is supported.
        """
        if isinstance(model,
                      _supported_layers) and not isinstance(model, _batch_norm + (nn.LayerNorm,)):
            weight = get_weight_sink([model])
            eq_indexes = EqualizationIndexes(0, weight[0].shape[0], 0)
            regions.append((model, eq_indexes))
        else:
            for module in model.children():
                self.find_module(module, regions)

    def setup(self):
        for region in self.regions:
            batch_dim = 0
            if hasattr(region, 'batch_first'):
                batch_dim = 0 if region.batch_first else 1

            hook_fn = partial(
                self.forward_stats_hook, name=region[0], batch_dim=batch_dim, use_inp=True)
            new_instance = KwargsForwardHook(region[0], hook_fn)
            ModuleInstanceToModuleInstance(region[0], new_instance).apply(self.model)
            self.hooks.append(new_instance)

    def apply(self, alpha):
        scale_factors = []
        self.remove_hooks()
        for region in self.regions:
            if self.float_act_map[region[0]] == None:
                continue
            sinks = region
            insert_mul_fn = partial(
                self.insert_mul_node, region=region[0], batch_dim=self.batch_dim_act_map[region[0]])
            scale_factors.append(
                _cross_layer_equalization({}, {'sinks0': sinks},
                                          False,
                                          scale_computation_type=self.scale_computation_type,
                                          list_of_act_val=[self.float_act_map[region[0]]],
                                          list_of_insert_mul_node_fn=[insert_mul_fn],
                                          alpha=alpha))
        return scale_factors

    def insert_mul_node(self, scale, shape, axis, region, batch_dim=0):
        mul_factor = self.create_mul_node(scale, shape, axis, batch_dim)
        rewriter = ModuleInstanceToModuleInstance(
            region, EqualizedModule(scale_module=mul_factor, layer=region))
        rewriter.apply(self.model)


class GraphActivationEqualization(ActivationEqualization):

    def __init__(
            self,
            model: GraphModule,
            add_mul_node: bool = False,
            scale_computation_type: str = 'maxabs'):
        super(GraphActivationEqualization, self).__init__(model, scale_computation_type)
        self.float_act_map = {}
        self.batch_dim_act_map = {}
        self.hooks = []
        self.add_mul_node = add_mul_node
        self.regions = _extract_regions(model, add_mul_node=add_mul_node, return_acts=True)

        if self.scale_computation_type == 'maxabs':
            self.scale_fn = _channel_maxabs
        elif self.scale_computation_type == 'range':
            self.scale_fn = _channel_range

    def setup(self):
        name_to_module = dict_name_to_module(self.model, self.regions)
        # Select only regions with activation to equalize through.
        # If a region has multiple scale varying activation, must also be dropped
        # because we can't propagate scaling factors
        regions_to_drop = []
        for region in self.regions:
            # This condition is for redudancy, since
            # a region with two scale-varying activations cannot be detected in the first place
            if len(region.acts) > 1 and any([isinstance(name_to_module[act_name],
                                                        _scale_varying_activations)
                                             for act_name in region.acts]):
                regions_to_drop.append(region)
                continue

            # We assume that the entire region has a unique batch_dim
            batch_dim = 0
            region_to_search = region.sinks if len(region.acts) == 0 else region.acts
            for name in region.srcs + region.sinks:
                module = name_to_module[name]
                if hasattr(module, 'batch_first'):
                    batch_dim = 0 if module.batch_first else 1
            for name in region_to_search:
                module = name_to_module[name]
                use_inp = True if region_to_search == region.sinks else False
                hook_fn = partial(
                    self.forward_stats_hook, name=name, batch_dim=batch_dim, use_inp=use_inp)
                new_instance = KwargsForwardHook(module, hook_fn)
                ModuleInstanceToModuleInstance(module, new_instance).apply(self.model)
                self.hooks.append(new_instance)

        self.regions = [x for x in self.regions if x not in regions_to_drop]

    def apply(self, alpha):
        scale_factors = []
        self.remove_hooks()
        name_to_module = dict_name_to_module(self.model, self.regions)
        for region in self.regions:
            region_to_search = region.sinks if len(region.acts) == 0 else region.acts
            if any([self.float_act_map[name] is None for name in region_to_search]):
                continue
            act_module = [name_to_module[act_name] for act_name in region.acts]
            list_of_act_val = [self.float_act_map[name] for name in region_to_search]
            sinks = [name_to_module[sink] for sink in region.sinks]
            # Filter out scale_varying activations from the srcs
            srcs = [
                name_to_module[src]
                for src in region.srcs
                if not isinstance(name_to_module[src], _scale_varying_activations)]

            list_of_insert_mul_node_fn = None
            if self.add_mul_node and any([
                    isinstance(act, _scale_varying_activations) for act in act_module]):
                # Even though we iterate, this list will always have a single element by definition
                list_of_insert_mul_node_fn = []
                for act_name in region.acts:
                    act_node = get_node(self.model, act_name)
                    list_of_insert_mul_node_fn.append(
                        partial(
                            self.insert_mul_node,
                            act_node=act_node,
                            batch_dim=self.batch_dim_act_map[act_name]))
            srcs_dict = _organize_region(region, name_to_module, 'srcs')
            sinks_dict = _organize_region(region, name_to_module, 'sinks')
            scale_factors.append(
                _cross_layer_equalization(
                    srcs_dict,
                    sinks_dict,
                    False,
                    scale_computation_type=self.scale_computation_type,
                    list_of_act_val=list_of_act_val,
                    list_of_insert_mul_node_fn=list_of_insert_mul_node_fn,
                    alpha=alpha))

        return scale_factors

    def insert_mul_node(self, scale, shape, axis, act_node, batch_dim=0):
        mul_factor = self.create_mul_node(scale, shape, axis, batch_dim)
        mul_factor_name = act_node.name + 'act_eq_mul'
        self.model.add_module(mul_factor_name, mul_factor)
        rewriter = InsertModuleCallAfter(mul_factor_name, act_node)
        rewriter.apply(self.model)
