# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from itertools import chain
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import warnings

import packaging
import packaging.version
import torch
from torch.fx import GraphModule as TorchGraphModule
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from brevitas import torch_version
from brevitas.fx import GraphModule
from brevitas.fx import Node
from brevitas.graph import ModuleToModuleByInstance
from brevitas.graph.base import GraphTransform
from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.base import ModuleInstanceRegisterParametrization
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import ModuleInstanceTransformTensor
from brevitas.graph.base import ModuleInstanceWrapModule
from brevitas.graph.base import Transform
from brevitas.graph.hadamard import find_closest_hadamard_number
from brevitas.graph.hadamard import get_hadK
from brevitas.graph.hadamard import matmul_hadU
from brevitas.graph.hadamard import matmul_hadU_cuda
from brevitas.graph.hadamard import random_hadamard_matrix
from brevitas.graph.utils import get_module
from brevitas.graph.utils import get_node
from brevitas.nn import ScaledDotProductAttention
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.equalized_layer import functional_rotate_input
from brevitas.nn.equalized_layer import INPUT_NAMES
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.quant_scale_bias import ScaleBias
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.utils.logging import setup_logger
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.parametrization_utils import ScaleWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import KwargsForwardHook
from brevitas.utils.torch_utils import pad_to_dim

logging = setup_logger(__name__)

# External optional dependency
try:
    # fast_hadamard_transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@main
    import fast_hadamard_transform
except:
    warnings.warn("fast_hadamard_transform package not found, using standard pytorch kernels")
    fast_hadamard_transform = None

# RMSNorm was introduced with torch 2.4
if torch_version >= packaging.version.parse('2.4'):
    RMSNorm = nn.RMSNorm
else:

    class PlaceholderRMSNorm:
        pass

    RMSNorm = PlaceholderRMSNorm

__all__ = [
    'GraphActivationEqualization',
    'LayerwiseActivationEqualization',
    'EqualizeGraph',
    'LayerwiseActivationRotation',
    'MergeLnAffine',
    'LayerNormToRMS',
    'GraphRotationEqualization']

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
    nn.functional.interpolate)

_select_op = (operator.getitem, operator.__getitem__)

_reshaping_op = ('view', 'reshape', 'flatten', 'contiguous', 'to', torch.reshape, torch.flatten)

_scale_varying_activations = (nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.GELU, nn.SiLU)

_residual_methods = ('add', 'add_')

_residual_fns = (torch.add, operator.add, operator.iadd, operator.__add__, operator.__iadd__)

_batch_norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

_ignore_ops = (getattr, 'size')


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


# Required for being hashable
@dataclass(eq=True, frozen=True)
class Region:
    srcs: Dict = field(default_factory=dict)
    sinks: Dict = field(default_factory=dict)
    acts: Tuple = field(default_factory=tuple)
    name_to_module: Dict = field(default_factory=dict)
    expand_region: bool = False

    @property
    def srcs_names(self):
        return [name.split("$")[0] for name in self.srcs.keys()]

    @property
    def sinks_names(self):
        return [name.split("$")[0] for name in self.sinks.keys()]

    def get_module_from_name(self, name: str) -> nn.Module:
        name = name.split("$")[0]
        return self.name_to_module[name]

    @property
    def max_shape_srcs(self):
        # Compute the number of output channel from the sources. If we are equalizing through cat,
        # we need to add together the number of channels. Otherwise, all sources must have the same
        # number of output channel.
        # Furthermore, all output channels of all the sources are always fully equalized.
        max_shape_srcs = 0
        for name, indexes in self.srcs.items():
            max_shape_srcs = max(max_shape_srcs, indexes.end + indexes.offset)
        return max_shape_srcs

    @property
    def max_shape_sinks(self):
        # Compute the number of input channel from the sinks. If we are equalizing through cat,
        # we need to slice and potentially select only a subset of input channel from sinks.
        max_shape_sinks = 0
        for name, indexes in self.sinks.items():
            max_shape_sinks = max(max_shape_sinks, indexes.offset + (indexes.end - indexes.start))
        return max_shape_sinks

    @property
    def is_valid(self):
        """
        To perform equalization, we need that the number of output channel of the sources matches the
        number of input channel of the sinks. If that's not the case, the region is considered invalid
        """
        # If max_shape_srcs is zero, it means we have no sources and only sinks
        # In this case, the region is considered valid since a standalone op will be added to counterbalance
        # this configuration.
        if self.max_shape_srcs == 0 and self.max_shape_sinks > 0:
            return True
        return self.max_shape_srcs == self.max_shape_sinks


@dataclass
class WalkRegionState:
    srcs: Dict = field(default_factory=dict)
    sinks: Dict = field(default_factory=dict)
    acts: Set = field(default_factory=set)
    history: set = field(default_factory=set)
    name_to_module: Dict = field(default_factory=dict)

    supported_srcs: set = _supported_layers
    supported_sinks: set = _supported_layers
    scale_invariant_function: set = _scale_invariant_op
    scale_invariant_layers: set = _scale_invariant_layers

    cat_encoutered: bool = False
    offset: int = 0
    update_offset: bool = False

    @property
    def srcs_names(self):
        return [name.split("$")[0] for name in self.srcs.keys()]

    @property
    def sinks_names(self):
        return [name.split("$")[0] for name in self.sinks.keys() if name is not _UNSUPPORTED_OP]

    def add(
            self,
            type: str,
            name: str,
            module: nn.Module,
            indexes: Optional[EqualizationIndexes] = None):
        if type == 'srcs' or type == 'sinks':
            assert indexes is not None
            full_source_name = name + '$' + str(indexes)
            getattr(self, type)[full_source_name] = indexes
        elif type == 'acts':
            self.acts.add(name)
        self.name_to_module[name] = module

    def add_srcs(self, src_name: str, src: nn.Module, indexes: EqualizationIndexes):
        self.add('srcs', src_name, src, indexes)

    def add_sinks(self, sink_name: str, sink: nn.Module, indexes: EqualizationIndexes):
        self.add('sinks', sink_name, sink, indexes)

    def add_acts(self, act_name: str, act: nn.Module):
        self.add('acts', act_name, act)

    def get_module_from_name(self, name: str) -> nn.Module:
        name = name.split("$")[0]
        return self.name_to_module[name]


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

    def __init__(
            self,
            model,
            alpha,
            add_mul_node=True,
            layerwise=True,
            enabled=True,
            blacklist_layers=None,
            co_optimize_act_weights=False,
            fuse_scaling=True) -> None:
        self.model = model
        self.alpha = alpha
        self.enabled = enabled
        self.add_mul_node = add_mul_node
        self.co_optimize_act_weights = co_optimize_act_weights
        self.fuse_scaling = fuse_scaling
        if layerwise:
            if not self.add_mul_node:
                raise ValueError("Layerwise activation equalization requires add_mul_node")
            self.graph_act_eq = LayerwiseActivationEqualization(
                self.model, blacklist_layers=blacklist_layers, fuse_scaling=self.fuse_scaling)
        else:
            if not isinstance(self.model, (TorchGraphModule, GraphModule)):
                raise TypeError(
                    "A Torch FX representation of the model is needed for Graph Activation Equalization"
                )
            self.graph_act_eq = GraphActivationEqualization(
                self.model,
                self.add_mul_node,
                co_optimize_act_weights=co_optimize_act_weights,
                fuse_scaling=self.fuse_scaling)
        self.scale_factors = None
        self.rewriters = None

    def __enter__(self):
        if self.enabled:
            self.graph_act_eq.setup()
        return self

    def __exit__(self, type, value, traceback):
        if self.enabled:
            self.scale_factors, self.rewriters = self.graph_act_eq.apply(self.alpha)


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
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
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
    elif isinstance(module,
                    (nn.Embedding, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return 1
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
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


def transpose(tensor: torch.Tensor, axis: int):
    """
    Given a tensor and an axis, this function re-arranges the tensor so that the axis and
    the first dimension are swapped.
    """
    shape = list(range(tensor.ndim))
    axis = shape[axis]
    shape.insert(0, axis)
    del shape[axis + 1]
    return tensor.permute(shape)


class EqualizationModuleWrapper:

    def __init__(
            self,
            module: nn.Module,
            weight_tensor_name: str,
            weight_axis: int,
            equalization_indexes: EqualizationIndexes,
            bias_tensor_name: Optional[str] = None,
            bias_axis: int = 0) -> None:
        self.module = module
        self._weight_tensor_name = weight_tensor_name
        self.weight_axis = weight_axis
        self.equalization_indexes = equalization_indexes
        self._bias_tensor_name = bias_tensor_name
        self.bias_axis = bias_axis

    @property
    def weight(self) -> Optional[nn.Parameter]:
        return getattr(self.module, self._weight_tensor_name, None)

    @property
    def bias(self) -> Optional[nn.Parameter]:
        if self._bias_tensor_name is not None:
            return getattr(self.module, self._bias_tensor_name, None)
        return None

    @abstractmethod
    def _get_transform_module_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_weight_range(self, scale_fn: Callable, **kwargs) -> torch.Tensor:
        pass

    def instantiate_rewriters(
            self,
            rewriter_class: Type[Union[ModuleInstanceTransformTensor,
                                       ModuleInstanceRegisterParametrization]],
            scaling_factor: Union[nn.Parameter, torch.Tensor]) -> List[Transform]:
        rewriters = []
        tensor_names_axis = [(self._weight_tensor_name, self.weight_axis)] + ([
            (self._bias_tensor_name, self.bias_axis)] if self.bias is not None else [])
        transform_module_kwargs = self._get_transform_module_kwargs()
        for tensor_name, axis in tensor_names_axis:
            rewriters.append(
                self._instantiate_rewriter_transform(
                    tensor_name=tensor_name,
                    rewriter_class=rewriter_class,
                    transform_module_class=ScaleWeightParametrization,
                    scaling_factor=scaling_factor,
                    axis=axis,
                    **transform_module_kwargs))
        return rewriters

    def _instantiate_rewriter_transform(
            self,
            tensor_name: str,
            rewriter_class: Type[Union[ModuleInstanceTransformTensor,
                                       ModuleInstanceRegisterParametrization]],
            transform_module_class: Type[nn.Module],
            **transform_module_kwargs) -> Transform:
        return rewriter_class(
            module=self.module,
            tensor_name=tensor_name,
            transform_module=transform_module_class(**transform_module_kwargs))


class EqualizationSourceWrapper(EqualizationModuleWrapper):

    def __init__(
            self,
            module: nn.Module,
            weight_tensor_name: str,
            weight_axis: int,
            equalization_indexes: EqualizationIndexes,
            bias_tensor_name: Optional[str] = None,
            bias_axis: int = 0) -> None:
        super().__init__(
            module,
            weight_tensor_name,
            weight_axis,
            equalization_indexes,
            bias_tensor_name,
            bias_axis)

    def _get_transform_module_kwargs(self) -> Dict[str, Any]:
        channel_start = self.equalization_indexes.offset + self.equalization_indexes.start
        channel_end = self.equalization_indexes.offset + self.equalization_indexes.end
        return {
            "start_end_idxs": None,
            "slice_idxs": (channel_start, channel_end),
            "use_inverse_scaling": False}

    # Determine the srcs_range based on where we are performing activation equalization or
    # weight equalization
    def get_weight_range(
            self,
            scale_fn: Callable,
            merge_bias: bool,
            bias_shrinkage: Optional[Union[float, str]] = None) -> torch.Tensor:
        weight = transpose(self.weight, self.weight_axis)
        if merge_bias:
            weight = _combine_weights_bias(weight, bias_shrinkage, self.bias)
        weight = weight.cpu().to(torch.float32)
        return scale_fn(weight.reshape(weight.size(0), -1))


class EqualizationSinkWrapper(EqualizationModuleWrapper):

    def __init__(
            self,
            module: nn.Module,
            weight_tensor_name: str,
            weight_axis: int,
            equalization_indexes: EqualizationIndexes) -> None:
        super().__init__(module, weight_tensor_name, weight_axis, equalization_indexes)

    def _get_transform_module_kwargs(self) -> Dict[str, Any]:
        channel_range = self.equalization_indexes.end - self.equalization_indexes.start
        return {
            "start_end_idxs": (self.equalization_indexes.start, self.equalization_indexes.end),
            "slice_idxs": (
                self.equalization_indexes.offset, self.equalization_indexes.offset + channel_range),
            "use_inverse_scaling": True}

    def get_weight_range(self, scale_fn: Callable) -> torch.Tensor:
        weight = transpose(self.weight.cpu().to(torch.float32), self.weight_axis)
        return scale_fn(weight.reshape(
            weight.size(0), -1))[self.equalization_indexes.start:self.equalization_indexes.end]


# When fuse_scaling = False, the scaling parameters are instances of nn.Parameter,
# which are registered to the scaling modules (used in the parametrization of the
# the weights). By default, these parameters have requires_grad set to True, and when
# registering the parametrizations, the forward pass of the parametrization modules
# is run (when unsafe is False, see _init_ of ParametrizationList), as well as when
# a module is part of multiple regions. Therefore, wrapping _cross_layer_equalization
# with torch.no_grad() prevents gradient functions (and gradient-related intermediate
# tensors) from being recorded when running this forward, thus preventing unnecessary
# memory consumption during the algorithm execution.
@torch.no_grad()
def _cross_layer_equalization(
        model: nn.Module,
        region: Region,
        merge_bias: bool,
        scale_computation_type: str,
        bias_shrinkage: Optional[Union[float, str]] = None,
        list_of_act_val: Optional[torch.Tensor] = None,
        list_of_insert_mul_node_fn: Optional[List[Callable]] = None,
        alpha: float = 0.5,
        co_optimize_act_weights: bool = False,
        fuse_scaling: bool = True) -> torch.Tensor:
    """
    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    rewriters = []

    # If equalization criteria are not met, we return a scalar one to indicate that no equalization
    # has been performed
    def _no_equalize():
        return torch.tensor(1., dtype=dtype), []

    # If a module has `allocate_params` attribute, we must load the weights following that method
    for name in (region.srcs_names + region.sinks_names):
        module = region.get_module_from_name(name)
        if hasattr(module, 'allocate_params'):
            module.allocate_params(module)

    act_sink_axes = {}
    act_sources_axes = {}
    single_module = region.get_module_from_name(next(iter(region.sinks_names)))
    device = next(single_module.parameters()).device
    dtype = next(single_module.parameters()).dtype

    # If region is not valid, don't equalize. If we are inserting a standalone mul, we don't need this check
    if not region.is_valid and list_of_insert_mul_node_fn is None:
        return _no_equalize()

    src_axes = {}
    for name, indexes in region.srcs.items():
        module = region.get_module_from_name(name)
        # If module is not supported, do not perform graph equalization
        weight_axis = _get_output_axis(module)
        act_sources_axes[name] = _get_act_axis(module)

        if isinstance(module, nn.MultiheadAttention):
            module = module.out_proj

        src_axes[name] = EqualizationSourceWrapper(
            module=module,
            weight_tensor_name="weight",
            weight_axis=weight_axis,
            equalization_indexes=indexes,
            bias_tensor_name="bias",
            bias_axis=0,
        )

    sink_axes = {}
    for name, indexes in region.sinks.items():
        module = region.get_module_from_name(name)
        weight_axis = _get_input_axis(module)
        act_sink_axes[name] = _get_act_axis(module)
        # For MultiheadAttention, we support only self-attention
        # For sinks, we only need to modify the weight but not the bias
        if isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
            # The weight attribute to equalize in nn.MultiheadAttention sinks is named "in_proj_weight"
            weight_tensor_name = "in_proj_weight"
        elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is None:
            return _no_equalize()
        else:
            weight_tensor_name = "weight"

        sink_axes[name] = EqualizationSinkWrapper(
            module=module,
            weight_tensor_name=weight_tensor_name,
            weight_axis=weight_axis,
            equalization_indexes=indexes,
        )

    # Check if any of the axis is None, which means that the module is not supported.
    # In that case, do not perform graph equalization
    axes_to_check = [m.weight_axis for m in list(src_axes.values()) + list(sink_axes.values())]
    if None in axes_to_check:
        return _no_equalize()

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

    scale_fn = _select_scale_computation_fn(scale_computation_type)
    srcs_range = -1 * torch.ones(region.max_shape_srcs, device='cpu', dtype=torch.float32)
    sinks_range = -1 * torch.ones(region.max_shape_sinks, device='cpu', dtype=torch.float32)
    for name, module in sink_axes.items():
        # Sinks can be partially equalized, thus we need to select
        # only the channels we are interested in
        indexes = region.sinks[name]
        # Compute the range of the channels we need to equalize
        weight_range = module.get_weight_range(scale_fn=scale_fn)
        # Compute the numbers of channels we are equalizing
        channel_range = indexes.end - indexes.start
        # Use the offset and the range to update the correct range in the sinks
        sinks_range[indexes.offset:indexes.offset + channel_range] = torch.max(
            sinks_range[indexes.offset:indexes.offset + channel_range], weight_range)

    for name, module in src_axes.items():
        # Srcs are always fully equalized, thus we simply need to apply the offset to position them
        # correctly with respect to the other srcs matrices.
        indexes = region.srcs[name]
        channel_start = indexes.offset + indexes.start
        channel_end = indexes.offset + indexes.end
        weight_range = module.get_weight_range(
            scale_fn=scale_fn, merge_bias=merge_bias, bias_shrinkage=bias_shrinkage)
        srcs_range[channel_start:channel_end] = torch.max(
            srcs_range[channel_start:channel_end], weight_range)
    if list_of_act_val is not None:
        list_of_act_val_shapes = [act_val.shape for act_val in list_of_act_val]
        if len(list_of_act_val_shapes) > 0:
            shape_0 = list_of_act_val_shapes[0]
            if any(shape_0 != shape for shape in list_of_act_val_shapes):
                return _no_equalize()
        list_of_act_val = [transpose(act_val, act_axis) for act_val in list_of_act_val]
        srcs_range_act = scale_fn(
            torch.cat([
                act_val.reshape(act_val.size(0), -1).cpu().to(torch.float32)
                for act_val in list_of_act_val],
                      1))

    if list_of_act_val is not None:
        if co_optimize_act_weights and len(src_axes) > 0:
            srcs_range = .5 * srcs_range + .5 * srcs_range_act
        else:
            srcs_range = srcs_range_act

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
        channelwise_no_equalize, torch.tensor(1., dtype=torch.float32, device='cpu'), sinks_range)
    srcs_range = torch.where(
        channelwise_no_equalize, torch.tensor(1., dtype=torch.float32, device='cpu'), srcs_range)

    srcs_range = torch.pow(srcs_range, alpha)
    sinks_range = torch.pow(sinks_range, 1 - alpha)
    scaling_factors = (sinks_range / srcs_range).to(device=device, dtype=dtype)
    # If the scaling factors are not fused, they are stored as a parameter instead
    if not fuse_scaling:
        scaling_factors = nn.Parameter(scaling_factors)

    if list_of_act_val is not None and list_of_insert_mul_node_fn is not None:
        for act_val_shape, insert_mul_node_fn in zip(list_of_act_val_shapes, list_of_insert_mul_node_fn):
            insert_mul_node_fn(scaling_factors, act_val_shape, act_axis)

    # Whether to apply the scaling in-place or parametrize the weights instead
    rewriter_class = ModuleInstanceTransformTensor if fuse_scaling else ModuleInstanceRegisterParametrization
    for module in chain(src_axes.values(), sink_axes.values()):
        rewriters.extend(module.instantiate_rewriters(rewriter_class, scaling_factors))

    for r in rewriters:
        model = r.apply(model)

    # If a module has `offload_params` attribute, we must offload the weights following that method
    for name in (region.srcs_names + region.sinks_names):
        module = region.get_module_from_name(name)
        if hasattr(module, 'offload_params'):
            module.offload_params(module)

    return scaling_factors, rewriters


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
    rewriters = []
    for i in range(iterations):
        scale_factor_max = None
        for region in regions:
            scale_factors_region, region_rewriters = _cross_layer_equalization(
                model,
                region,
                merge_bias=merge_bias,
                bias_shrinkage=bias_shrinkage,
                scale_computation_type=scale_computation_type)
            scale_factor_region_max = torch.max(torch.abs(1 - scale_factors_region))
            rewriters.extend(region_rewriters)
            if scale_factor_max is not None:
                scale_factor_max = torch.max(scale_factor_max, scale_factor_region_max)
            else:
                scale_factor_max = scale_factor_region_max
        if threshold is not None and scale_factor_max < threshold:
            break
    return model


def _is_supported_module(
        graph_model: GraphModule, node: Node, supported_layers: Set = _supported_layers) -> bool:
    if node.op == 'call_module':
        module = get_module(graph_model, node.target)
        if isinstance(module, supported_layers):
            # We support only self-attention
            if isinstance(module, nn.MultiheadAttention):
                kwargs = dict(node.kwargs)
                # When using hf/accelerate, we need to check the signature of the original forward
                forward_to_check = module._old_forward if hasattr(
                    module, '_old_forward') else module.forward
                kwargs.update(zip(forward_to_check.__code__.co_varnames[1:], node.args))
                return kwargs['query'].name == kwargs['key'].name == kwargs['value'].name
            return True
    return False


def _is_scale_invariant_module(
        graph_model: GraphModule,
        node: Node,
        scale_invariant_layers=_scale_invariant_layers) -> bool:
    return node.op == 'call_module' and isinstance(
        get_module(graph_model, node.target), scale_invariant_layers)


def _is_scale_varying_activation(graph_model, node):
    node_target = node.meta.get('orig_target', node.target)
    return node.op == 'call_module' and isinstance(
        get_module(graph_model, node_target), _scale_varying_activations)


def _is_scale_invariant_function(node: Node, scale_invariant_op: Set = _scale_invariant_op) -> bool:
    node_target = node.meta.get('orig_target', node.target)
    out = node.op in (
        'call_function',
        'call_method') and node_target in scale_invariant_op + _select_op + _reshaping_op
    if node_target == nn.functional.interpolate:
        out &= node.kwargs.get('mode', None) == 'nearest'
    return out


def get_weight_source(module):
    transpose = lambda weight, axis: weight if axis == 0 else weight.transpose(0, 1)
    if isinstance(module, nn.MultiheadAttention) and not hasattr(module, 'out_proj'):
        raise RuntimeError("Configuration for Multiheadattention not supported")
    weight = module.out_proj.weight if isinstance(module, nn.MultiheadAttention) else module.weight
    axis = _get_output_axis(module)
    weight = transpose(weight, axis)
    return weight


def get_weight_sink(module):
    transpose = lambda weight, axis: weight if axis == 0 else weight.transpose(0, 1)
    if isinstance(module, nn.MultiheadAttention) and not hasattr(module, 'in_proj_weight'):
        raise RuntimeError("Configuration for Multiheadattention not supported")
    weight = module.in_proj_weight if isinstance(module, nn.MultiheadAttention) else module.weight
    axis = _get_input_axis(module)
    weight = transpose(weight, axis)
    return weight


def find_srcs_channel_dim(state, model, inp_node):
    inp_node_target = inp_node.meta.get('orig_target', inp_node.target)
    if _is_supported_module(model, inp_node, state.supported_srcs):
        # If we meet a supported module, determine the channel shape
        module = get_module(model, inp_node_target)
        # Since we are walking up, we consider the module as srcs
        weight = get_weight_source(module)
        channel = weight.shape[0]
        return channel
    elif _is_add(inp_node):
        all_channels = []
        for n in inp_node.all_input_nodes:
            all_channels.append(find_srcs_channel_dim(state, model, n))
        # All branches to add should have the same amount of channels
        if all([channel == all_channels[0] for channel in all_channels]):
            return all_channels[0]
        else:
            return _UNSUPPORTED_OP
    elif _is_cat(inp_node):
        total_channels = 0
        # If it's cat, we need to sum the channel shape of all the branches
        for n in inp_node.all_input_nodes:
            total_channels += find_srcs_channel_dim(state, model, n)
        return total_channels
    elif _is_scale_invariant_module(model, inp_node,
                                    state.scale_invariant_layers) or _is_scale_invariant_function(
                                        inp_node, state.scale_invariant_function):
        return find_srcs_channel_dim(state, model, inp_node.all_input_nodes[0])
    else:
        return _UNSUPPORTED_OP


def cat_handler(graph_model: GraphModule, starting_node: Node, state: WalkRegionState):

    state.srcs.clear()
    state.sinks.clear()
    state.history.clear()
    # Keep track that concatenation has been encoutered once
    state.cat_encoutered = True
    state.update_offset = True
    state.offset = 0
    find_srcs(graph_model, starting_node, state)
    state.update_offset = False
    state.offset = 0
    find_sinks(graph_model, starting_node, state)


def _is_cat(node):
    node_target = node.meta.get('orig_target', node.target)
    return node_target in (torch.cat,)


def _is_add(node):
    node_target = node.meta.get('orig_target', node.target)
    return (
        node.op == 'call_method' and node_target in _residual_methods or
        node.op == 'call_function' and node_target in _residual_fns)


def find_srcs(graph_model: GraphModule, starting_node: Node,
              state: WalkRegionState) -> Dict[str, Set]:
    node_list = starting_node.all_input_nodes
    update_offset_state = state.update_offset
    for node in node_list:
        # we keep a history of how the graph has been walked already, invariant to the direction,
        # to avoid getting stuck in a loop
        node_target = node.meta.get('orig_target', node.target)
        path = (node, starting_node)
        if path not in state.history:
            state.history.add(path)
        else:
            continue
        if _is_supported_module(graph_model, node, state.supported_srcs):
            module = get_module(graph_model, node_target)
            weight = get_weight_source(module)
            eq_indexes = EqualizationIndexes(0, weight.shape[0], state.offset)

            # After we found a source, we need to check if it branches into multiple sinks
            state.add_srcs(node_target, module, eq_indexes)
            find_sinks(graph_model, node, state)
            state.offset = state.offset if not state.update_offset else state.offset + weight.shape[
                0]
        elif _is_scale_invariant_module(
                graph_model, node, state.scale_invariant_layers) or _is_scale_invariant_function(
                    node, state.scale_invariant_function):
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
        elif (node.op == 'call_method' and node_target in _residual_methods or
              node.op == 'call_function' and node_target in _residual_fns):
            state.update_offset = False
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
            state.update_offset = update_offset_state
        elif _is_cat(node):
            # The first time we encoutered a cat differes from all subsequent ones
            if not state.cat_encoutered:
                # We restart the region search starting from the cat
                cat_handler(graph_model, node, state)
            else:
                state.update_offset = False
                find_sinks(graph_model, node, state)
                state.update_offset = True
                find_srcs(graph_model, node, state)
                state.update_offset = update_offset_state
        elif node_target in _ignore_ops:
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
        node_target = node.meta.get('orig_target', node.target)
        path = (starting_node, node)
        if path not in state.history:
            state.history.add(path)
        else:
            continue
        if _is_supported_module(graph_model, node, state.supported_sinks):
            module = get_module(graph_model, node_target)
            weight = get_weight_sink(module)
            eq_indexes = EqualizationIndexes(0, weight.shape[0], state.offset)
            state.add_sinks(node_target, module, eq_indexes)

        elif _is_scale_invariant_module(
                graph_model, node, state.scale_invariant_layers) or _is_scale_invariant_function(
                    node, state.scale_invariant_function):
            find_sinks(graph_model, node, state)
        elif (node.op == 'call_method' and node_target in _residual_methods or
              node.op == 'call_function' and node_target in _residual_fns):
            state.update_offset = False
            find_sinks(graph_model, node, state)
            find_srcs(graph_model, node, state)
            state.update_offset = update_offset_state
        elif _is_cat(node):
            # The first time we encoutered a cat differes from all subsequent ones
            if not state.cat_encoutered:
                # We restart the region search starting from the cat
                cat_handler(graph_model, node, state)
            else:
                # In this case we define all our sinks, and isolate only the channels we want
                # to equalize (start, end).
                # Furthermore, we need to consider the offset given by the sources of the second cat
                index = node.all_input_nodes.index(starting_node)
                channels = []
                for n in node.all_input_nodes:
                    channel_dim = find_srcs_channel_dim(state, graph_model, n)
                    channels.append(channel_dim)

                # If we found an unsupported op while walking up, we exit this branch and
                # invalidate the region
                if _UNSUPPORTED_OP in channels:
                    state.sinks[_UNSUPPORTED_OP] = _UNSUPPORTED_OP
                    continue
                start = sum(channels[:index])
                end = start + channels[index]
                new_state = WalkRegionState(offset=state.offset)
                find_sinks(graph_model, node, new_state)

                for k in new_state.sinks_names:
                    state.add_sinks(
                        k,
                        new_state.get_module_from_name(k),
                        EqualizationIndexes(start, end, new_state.offset))
                state.srcs.update(new_state.srcs)
        elif node_target in _ignore_ops:
            continue
        else:
            # If we meet an unrecognized op, we add None to invalidate the region
            state.sinks[_UNSUPPORTED_OP] = _UNSUPPORTED_OP


def _extract_regions(
        graph_model: GraphModule,
        add_mul_node: bool = False,
        return_acts: bool = False,
        state_impl_kwargs=None) -> List[Region]:
    regions = list()
    for node in graph_model.graph.nodes:
        if state_impl_kwargs is not None:
            state = WalkRegionState(**state_impl_kwargs)
        else:
            state = WalkRegionState()
        if _is_supported_module(graph_model, node, state.supported_srcs) or (
                add_mul_node and _is_scale_varying_activation(graph_model, node)):
            if _is_scale_varying_activation(graph_model, node):
                module = get_module(graph_model, node.target)
                state.add_acts(node.target, module)
            else:
                module = get_module(graph_model, node.target)
                weight = get_weight_source(module)
                eq_indexes = EqualizationIndexes(0, weight.shape[0], 0)
                state.add_srcs(node.target, module, eq_indexes)
            find_sinks(graph_model, node, state)
            if len(state.sinks) > 0 and _UNSUPPORTED_OP not in state.sinks.keys():
                sorted_srcs = dict(sorted(state.srcs.items()))
                sorted_sinks = dict(sorted(state.sinks.items()))
                sorted_acts = tuple(sorted(state.acts))
                if return_acts:
                    region = Region(
                        srcs=sorted_srcs,
                        sinks=sorted_sinks,
                        acts=sorted_acts,
                        name_to_module=state.name_to_module)
                else:
                    region = Region(
                        srcs=sorted_srcs, sinks=sorted_sinks, name_to_module=state.name_to_module)

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
        # It is not possible to equalize through LayerNorm/BatchNorm as sink
        supported_sinks = tuple([
            x for x in _supported_layers if x not in (nn.LayerNorm, *_batch_norm)])
        regions = _extract_regions(
            graph_model, state_impl_kwargs={'supported_sinks': supported_sinks})
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
        if isinstance(scale, nn.Parameter):
            # Register as parameter, thus potentially being tied with the scaling factors
            # kept by other modules
            mul_factor.weight = scale
        else:
            # Modify in-place the values of the weight parameter
            mul_factor.weight.data = scale
        return mul_factor

    def forward_stats_hook(self, module, *args, name, batch_dim=0, use_inp=True, **kwargs):
        # Check for MHA Cross attention, and if found, skip it
        # When using hf/accelerate, we need to check the signature of the original forward
        forward_to_check = module._old_forward if hasattr(
            module, '_old_forward') else module.forward
        kwargs.update(zip(forward_to_check.__code__.co_varnames[1:], args[:-1]))
        if 'query' in kwargs and 'key' in kwargs and 'value' in kwargs:
            if kwargs['query'].data_ptr() != kwargs['key'].data_ptr() != kwargs['value'].data_ptr():
                self.float_act_map[name] = None
                return

        input_kwarg = [x for x in kwargs.keys() if x in INPUT_NAMES][0]
        if use_inp:
            x = kwargs[input_kwarg]
        elif not use_inp:
            x = args[-1]

        # Extra check for batch_dim
        if hasattr(x, 'names') and 'N' in x.names:
            batch_dim = x.names.index('N')

        self.batch_dim_act_map[name] = batch_dim

        dtype = x.dtype
        input_scales = self.scale_fn(x.to(torch.float32), dim=batch_dim).to(dtype)
        if name not in self.float_act_map:
            self.float_act_map[name] = input_scales
        else:
            self.float_act_map[name] = torch.max(self.float_act_map[name], input_scales)

    def remove_hooks(self):
        for hook in self.hooks:
            ModuleInstanceToModuleInstance(hook, hook.module).apply(self.model)


class LayerwiseActivationEqualization(ActivationEqualization):

    def __init__(
            self,
            model,
            scale_computation_type: str = 'maxabs',
            blacklist_layers: Optional[List[str]] = None,
            fuse_scaling: bool = True):
        super(LayerwiseActivationEqualization, self).__init__(model, scale_computation_type)
        self.float_act_map = {}
        self.batch_dim_act_map = {}
        self.hooks = []
        self.add_mul_node = True
        self.blacklist_layers = blacklist_layers
        self.fuse_scaling = fuse_scaling

        regions: List[Region] = []
        self.find_module(model, regions)
        self.regions = regions

        if self.scale_computation_type == 'maxabs':
            self.scale_fn = _channel_maxabs
        elif self.scale_computation_type == 'range':
            self.scale_fn = _channel_range

    def find_module(self, model, regions: List, prefix=''):
        """
        Iterate through the model looking at immediate children of every module to look for supported modules.
        This allows us to stop the search when we meet a top-level module that is supported.
        """
        if isinstance(model,
                      _supported_layers) and not isinstance(model, _batch_norm + (nn.LayerNorm,)):
            if self.blacklist_layers is not None and prefix in self.blacklist_layers:
                return
            weight = get_weight_sink(model)
            eq_indexes = EqualizationIndexes(0, weight.shape[0], 0)
            region = Region(sinks={prefix: eq_indexes}, name_to_module={prefix: model})
            regions.append(region)
        else:
            for name, module in model.named_children():
                full_name = prefix + '.' + name if prefix != '' else name
                self.find_module(module, regions, full_name)

    def setup(self):
        for region in self.regions:
            name = list(region.sinks.keys())[0]
            module = region.get_module_from_name(name)
            batch_dim = 0
            if hasattr(region, 'batch_first'):
                batch_dim = 0 if region.batch_first else 1

            hook_fn = partial(
                self.forward_stats_hook, name=module, batch_dim=batch_dim, use_inp=True)
            new_instance = KwargsForwardHook(module, hook_fn)
            ModuleInstanceToModuleInstance(module, new_instance).apply(self.model)
            self.hooks.append(new_instance)

    def apply(self, alpha):
        scale_factors = []
        rewriters = []
        self.remove_hooks()
        for region in self.regions:
            name = list(region.sinks.keys())[0]
            module = region.get_module_from_name(name)
            if self.float_act_map.get(module, None) == None:
                logging.info(f"Module {name} not found during layerwise activation equalization")
                continue
            insert_mul_fn = partial(
                self.insert_mul_node, region=module, batch_dim=self.batch_dim_act_map[module])
            scale_factors_region, rewriters_region = _cross_layer_equalization(
                self.model,
                region,
                False,
                scale_computation_type=self.scale_computation_type,
                list_of_act_val=[self.float_act_map[module]],
                list_of_insert_mul_node_fn=[insert_mul_fn],
                alpha=alpha,
                fuse_scaling=self.fuse_scaling)
            scale_factors.append(scale_factors_region)
            rewriters.extend(rewriters_region)
        return scale_factors, rewriters

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
            scale_computation_type: str = 'maxabs',
            co_optimize_act_weights: bool = False,
            fuse_scaling: bool = True):
        super(GraphActivationEqualization, self).__init__(model, scale_computation_type)
        self.float_act_map = {}
        self.batch_dim_act_map = {}
        self.hooks = []
        self.hooked_modules = set()
        self.add_mul_node = add_mul_node
        self.co_optimize_act_weights = co_optimize_act_weights
        self.fuse_scaling = fuse_scaling

        # It is not possible to equalize through LayerNorm/BatchNorm as sink
        supported_sinks = tuple([
            x for x in _supported_layers if x not in (nn.LayerNorm, *_batch_norm)])
        self.regions = _extract_regions(
            model,
            add_mul_node=add_mul_node,
            return_acts=True,
            state_impl_kwargs={'supported_sinks': supported_sinks})

        if self.scale_computation_type == 'maxabs':
            self.scale_fn = _channel_maxabs
        elif self.scale_computation_type == 'range':
            self.scale_fn = _channel_range

    def setup(self):
        # Select only regions with activation to equalize through.
        # If a region has multiple scale varying activation, must also be dropped
        # because we can't propagate scaling factors
        regions_to_drop = []
        for region in self.regions:
            # This condition is for redudancy, since
            # a region with two scale-varying activations cannot be detected in the first place
            if len(region.acts) > 1 and any([isinstance(region.get_module_from_name(act_name),
                                                        _scale_varying_activations)
                                             for act_name in region.acts]):
                regions_to_drop.append(region)
                continue

            # We assume that the entire region has a unique batch_dim
            batch_dim = 0
            for name in region.srcs:
                module = region.get_module_from_name(name)
                if hasattr(module, 'batch_first') and not module.batch_first:
                    batch_dim = 1
            for name in region.sinks:
                module = region.get_module_from_name(name)
                if hasattr(module, 'batch_first') and not module.batch_first:
                    batch_dim = 1

            region_to_search = region.sinks_names if len(region.acts) == 0 else region.acts
            for name in region_to_search:
                module = region.get_module_from_name(name)
                if module not in self.hooked_modules:
                    self.hooked_modules.add(module)
                    use_inp = True if region_to_search == region.sinks_names else False
                    hook_fn = partial(
                        self.forward_stats_hook, name=name, batch_dim=batch_dim, use_inp=use_inp)
                    new_instance = KwargsForwardHook(module, hook_fn)
                    ModuleInstanceToModuleInstance(module, new_instance).apply(self.model)
                    self.hooks.append(new_instance)

        self.regions = [x for x in self.regions if x not in regions_to_drop]

    def apply(self, alpha):
        scale_factors = []
        rewriters = []
        self.remove_hooks()
        for region in self.regions:
            region_names = region.sinks_names if len(region.acts) == 0 else region.acts
            if any([self.float_act_map[name] is None for name in region_names]):
                continue
            act_module = [region.get_module_from_name(act_name) for act_name in region.acts]
            list_of_act_val = [self.float_act_map[name] for name in region_names]

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

            scale_factors_region, rewriters_region = _cross_layer_equalization(
                self.model,
                region,
                False,
                scale_computation_type=self.scale_computation_type,
                list_of_act_val=list_of_act_val,
                list_of_insert_mul_node_fn=list_of_insert_mul_node_fn,
                alpha=alpha,
                co_optimize_act_weights=self.co_optimize_act_weights,
                fuse_scaling=self.fuse_scaling)
            scale_factors.append(scale_factors_region)
            rewriters.append(rewriters_region)

        return scale_factors, rewriters

    def insert_mul_node(self, scale, shape, axis, act_node, batch_dim=0):
        mul_factor = self.create_mul_node(scale, shape, axis, batch_dim)
        mul_factor_name = act_node.name + 'act_eq_mul'
        self.model.add_module(mul_factor_name, mul_factor)
        rewriter = InsertModuleCallAfter(mul_factor_name, act_node)
        rewriter.apply(self.model)


def _apply_had_device(tensor, had_K, K):
    is_cuda = 'cuda' in str(tensor.device) and torch.version.cuda is not None
    # Accelerated kernel only available for CUDA
    if is_cuda and fast_hadamard_transform is not None:
        return matmul_hadU_cuda(tensor, had_K, K)
    else:
        return matmul_hadU(tensor)


def _apply_ort_device(tensor, ort, *args):
    ort = ort.type_as(tensor)
    return torch.matmul(tensor, ort)


# Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/eval_utils/rotation_utils.py#L26
def random_orthogonal_matrix(size):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0).float()
    return q


def _apply_rotate(
        model: nn.Module,
        regions: List[Region],
        full_rotation_method='had',
        fuse_rotations: bool = True,
        apply_inplace_rotations: bool = True):
    rewriters = []
    # First, rotations on orphan sinks are applied so the order in which rotations are
    # applied is consistent, irrespective of the value of fuse_rotations. This is due to
    # the fact that parametrizations need to be registered, once all the in-place
    # operations have taken place
    regions = [region for region in regions if len(region.srcs) == 0] + [
        region for region in regions if len(region.srcs) > 0]

    # Pre-initialize to None to avoid issue down the line
    expanded_rot_mat, expanded_K, rot_mat, K = None, None, None, None
    for region in regions:
        insert_rotation_module = len(region.srcs) == 0
        if not region.is_valid:
            logging.info(f"Region not valid, skipping it")
        hidden_dim = region.max_shape_sinks
        if not insert_rotation_module and full_rotation_method == 'ort':
            rot_mat = random_orthogonal_matrix(hidden_dim)
            rot_func = _apply_ort_device
        elif not insert_rotation_module and not fuse_rotations:
            # If the model is distributed across GPUs, the device will be
            # not be the same for all of the parameters, so explicit moves
            # to the same device as the weights need to be added
            device = next(model.parameters()).device
            rot_mat = random_hadamard_matrix(hidden_dim, device)
            rot_func = _apply_ort_device
        else:
            try:
                # Build hadamard rotation matrix
                rot_mat, K = get_hadK(hidden_dim)
                hidden_dim = find_closest_hadamard_number(hidden_dim)
                expanded_rot_mat, expanded_K = get_hadK(int(hidden_dim))
                rot_func = _apply_had_device
            except AssertionError as e:
                logging.info(f"Incompatible dim {hidden_dim} for hadamard rotation")
                if not insert_rotation_module:
                    logging.info("Falling back to orthogonal matrices")
                    rot_mat = random_orthogonal_matrix(hidden_dim)
                    rot_func = _apply_ort_device
                else:
                    logging.info("Skipping region")
                    continue

        # Cast rotation matrix to the weight dtype
        if rot_mat is not None:
            dtype = next(model.parameters()).dtype
            rot_mat = rot_mat.to(dtype=dtype)
        # If the rotation is not fused, redefine as a Parameter, to enable its optimization
        if not insert_rotation_module and not fuse_rotations:
            rot_mat = torch.nn.Parameter(rot_mat)

        for name, indexes in region.srcs.items():
            module = region.get_module_from_name(name)
            # Rotate "bias" if present
            tensor_names_axis = [("weight", _get_output_axis(module))] + ([
                ("bias", 1)] if getattr(module, 'bias', None) is not None else [])
            # If rotations are fused, transform is applied directly onto the tensor
            rewriter_class = ModuleInstanceTransformTensor if fuse_rotations else ModuleInstanceRegisterParametrization
            # Obtain rewriters for applying the rotations
            for tensor_name, axis in tensor_names_axis:
                rewriter = rewriter_class(
                    module=module,
                    tensor_name=tensor_name,
                    transform_module=RotationWeightParametrization(
                        rot_mat=rot_mat,
                        rot_func=rot_func,
                        axis=axis,
                        K=K,
                    ))
                rewriters.append(rewriter)

        for name, indexes in region.sinks.items():
            module = region.get_module_from_name(name)
            weight_axis = _get_input_axis(module)

            # Only "weight" is rotated
            if region.expand_region:
                rot_mat, K = expanded_rot_mat, expanded_K
                assert isinstance(module, nn.Linear), "Currently only Linear layers support expanded hadamard"
                hidden_dim = module.weight.shape[1]
                new_hidden = find_closest_hadamard_number(hidden_dim)
                new_weights = pad_to_dim(module.weight.data, weight_axis, new_hidden)
                # Modify the weights in-place
                setattr(module, 'weight', torch.nn.Parameter(new_weights))
                module.in_features = int(new_hidden)

            # If rotations are fused or if the module is an orphan sink, transform is applied directly onto the tensor
            rewriter_class = ModuleInstanceTransformTensor if insert_rotation_module or fuse_rotations else ModuleInstanceRegisterParametrization
            # Obtain rewriters for applying the rotations
            rewriter = rewriter_class(
                module=module,
                tensor_name='weight',
                transform_module=RotationWeightParametrization(
                    rot_mat=rot_mat,
                    rot_func=rot_func,
                    axis=weight_axis,
                    K=K,
                ))
            rewriters.append(rewriter)
            # Replace by RotatedModule in orphan sink
            if insert_rotation_module and len(region.srcs) == 0:
                rewriter = ModuleInstanceWrapModule(
                    module,
                    RotatedModule,
                    "layer", {
                        "had_mat": rot_mat, "k": K, "expand": region.expand_region})
                rewriters.append(rewriter)
    if apply_inplace_rotations:
        for r in rewriters:
            # The parametrizations need to be registered after the potential HF hooks have been
            # removed, as otherwise the device maps will not match the structure of the
            # model's state_dict after the registration of the parametrizations.
            if not isinstance(r, ModuleInstanceRegisterParametrization):
                model = r.apply(model)
    return rewriters


# This function is adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py
def _untie_parameters_with_parametrizations(model: torch.nn.Module):
    # get ALL model parameters and their names
    all_named_parameters = {
        name: param for name, param in model.named_parameters(remove_duplicate=False)}

    # get ONLY unique named parameters,
    # if parameter is tied and have multiple names, it will be included only once
    no_duplicate_named_parameters = {
        name: param for name, param in model.named_parameters(remove_duplicate=True)}

    # the difference of the two sets will give us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    for tied_param_name in tied_param_names:
        tied_param_name_split = tied_param_name.split(".")
        # The names of the original parameters after registering the parametrization
        # have the format "prefix.parametrizations.tensor_name.original", e.g.
        # "model.layer.parametrizations.weight.original". This allows to identify
        # which subset of tied parameters are original tied parameters of the module
        if len(tied_param_name_split) >= 3 and tied_param_name_split[
                -3] == "parametrizations" and tied_param_name_split[-1] == "original":
            # If that is the case, retrieve the parent module
            parent_module = recurse_getattr(model, ".".join(tied_param_name_split[:-1]))
            # And set to a new parameter, thus breaking the tie
            setattr(parent_module, "original", nn.Parameter(all_named_parameters[tied_param_name]))

    return model


def fuse_parametrizations(model: nn.Module) -> nn.Module:
    # First of all, parameters that have parametrizations need to be untied
    model = _untie_parameters_with_parametrizations(model)
    # Then, parametrizations can be safely removed
    for module in model.modules():
        if parametrize.is_parametrized(module):
            # Names of the tensors that can potentially be parametrized
            tensor_names = list(module.parametrizations.keys())
            # Remove parametrizations from each tensor
            for tensor_name in tensor_names:
                if parametrize.is_parametrized(module) and tensor_name in module.parametrizations:
                    # Check if the module has any quantization-related children
                    state_dict = None
                    for submodule in module.modules():
                        if isinstance(submodule,
                                      (WeightQuantProxyFromInjector, BiasQuantProxyFromInjector)):
                            state_dict = submodule.state_dict()
                            break
                    # The rotated tensor is saved by setting leave_parametrized=True
                    parametrize.remove_parametrizations(
                        module, tensor_name, leave_parametrized=True)
                    # Restore the state of the quantization modules, as these might have been reset
                    # when registering the parametrized parameter
                    if state_dict is not None:
                        submodule.load_state_dict(state_dict)
    return model


def _replace_bias(next_module, new_bias):
    new_bias = new_bias.view(-1)
    if next_module.bias is not None:
        next_module.bias.data.copy_(new_bias)
    else:
        new_bias = new_bias.to(next_module.weight.device).to(next_module.weight.dtype)
        next_module.register_parameter('bias', nn.Parameter(new_bias))


def _merge_ln(layer_norm, next_module, scale_bias_by_weight):
    view_shape = (1, -1)
    # Merge weight
    if scale_bias_by_weight and hasattr(layer_norm, 'bias'):
        layer_norm.bias.data /= layer_norm.weight.data
    # We can't do an inplace update as some layers we merge into like lm_head might share the weight tensor
    scale = layer_norm.weight.data.view(view_shape).expand_as(next_module.weight)
    next_module.weight = nn.Parameter(next_module.weight.clone() * scale)

    # Merge bias, new_bias includes the bias of next_module by going through its fwd
    if hasattr(layer_norm, 'bias'):
        inp = layer_norm.bias.data.view(view_shape)
        new_bias = next_module(inp)
        _replace_bias(next_module, new_bias)


class RotationEqualization(GraphTransform):

    def __init__(self, blacklist_layers, layers_to_expand) -> None:
        super(RotationEqualization, self).__init__()
        if blacklist_layers is not None:
            self.blacklist_layers = blacklist_layers
        else:
            self.blacklist_layers = []
        if layers_to_expand is not None:
            self.layers_to_expand = layers_to_expand
        else:
            self.layers_to_expand = []
        self.supported_sinks = ()

    def find_module(
            self,
            model: nn.Module,
            regions: List[Region],
            prefix: str = '',
            blacklist_layers: Optional[List[str]] = None):
        """
        Iterate through the model looking at immediate children of every module to look for supported modules.
        This allows us to stop the search when we meet a top-level module that is supported.
        """
        if isinstance(model, self.supported_sinks):
            if prefix in blacklist_layers:
                return
            weight = get_weight_sink(model)
            eq_indexes = EqualizationIndexes(0, weight.shape[0], 0)
            region = Region(sinks={'sinks0': eq_indexes}, name_to_module={'sinks0': model})
            regions.append(region)
        else:
            for name, module in model.named_children():
                full_name = prefix + '.' + name if prefix != '' else name
                self.find_module(module, regions, full_name, blacklist_layers)

    def find_module_by_name(self, model: nn.Module, regions: List[Region], prefix: str = ''):
        """
        Iterate through the model looking at immediate children of every module to look for named modules.
        This allows us to stop the search when we meet a top-level module that is supported.
        """
        if prefix in self.layers_to_expand:
            if prefix in self.blacklist_layers:
                return
            weight = get_weight_sink(model)
            eq_indexes = EqualizationIndexes(0, weight.shape[0], 0)
            region = Region(
                sinks={'sinks0': eq_indexes}, name_to_module={'sinks0': model}, expand_region=True)
            regions.append(region)
        else:
            for name, module in model.named_children():
                full_name = prefix + '.' + name if prefix != '' else name
                self.find_module_by_name(module, regions, full_name)


class GraphRotationEqualization(RotationEqualization):

    def __init__(
            self,
            blacklist_layers: Optional[List[str]] = None,
            orphan_sink: bool = False,
            sdpa_regions: bool = False,
            rotate_matmul: bool = False,
            use_parametrized_rotations: bool = False,
            full_rotation_method: str = 'had',
            layers_to_expand: Optional[List[str]] = None,
            return_rewriters: bool = False) -> None:
        super(GraphRotationEqualization, self).__init__(blacklist_layers, layers_to_expand)

        self.supported_srcs = (nn.Linear, nn.Embedding)
        self.supported_sinks = (nn.Linear)
        common_scale_invariant = list(_scale_invariant_layers)
        common_scale_invariant.remove(torch.nn.ReLU)
        common_scale_invariant.remove(torch.nn.LeakyReLU)
        self.scale_invariant_layers = tuple(common_scale_invariant) + (RMSNorm,)
        self.scale_invariant_function = ()
        self.orphan_sink = orphan_sink
        self.rotate_matmul = rotate_matmul
        self.full_rotation_method = full_rotation_method
        self.return_rewriters = return_rewriters
        self.sdpa_regions = sdpa_regions
        if use_parametrized_rotations:
            # NOTE: When use_parametrized_rotations=False, parametrized rotations are applied. This changes the attribute __class__
            # of the parametrized module, e.g. to"<class 'torch.nn.utils.parametrize.ParametrizedLinear'>".
            # Therefore, algorithms that do type checking might need to use type_before_parametrizations(module),
            # instead of only type(module) (see layerwise_layer_handler). Algorithms that rely on in-place modifications
            # of the weights should not operate on parametrized modules. In this situation, parametrizations
            # need to be removed beforehand by invoking fuse_parametrizations
            warnings.warn(
                "Using parametrized results might break type-checking, which could lead to unexpected behaviour."
            )
        self.use_parametrized_rotations = use_parametrized_rotations

    def rotate_matmuls(self, graph_module):
        matmul_nodes = list(graph_module.graph.nodes)
        matmul_nodes = [c for c in matmul_nodes if c.name == 'matmul']
        for node in matmul_nodes:
            with graph_module.graph.inserting_before(node):
                matmul_arg0 = graph_module.graph.call_function(
                    functional_rotate_input, args=(node.args[0],))
                matmul_arg1 = graph_module.graph.call_function(
                    functional_rotate_input, args=(node.args[1],), kwargs={'transpose': True})
            args = list(node.args)
            args[0] = matmul_arg0
            args[1] = matmul_arg1
            node.args = tuple(args)

            graph_module.recompile()
            graph_module.graph.lint()

    def rotate_sdpa(self, graph_module):
        sdpa_nodes = list(graph_module.graph.nodes)
        sdpa_nodes = [
            c for c in sdpa_nodes
            if ('scaled_dot_product' in str(c.meta.get('orig_target', c.target)))]
        regions = []

        def find_src(node):
            if node.op != 'call_module':
                return find_src(node.args[0])
            else:
                return node

        def find_sink(node):
            output_node = list(node.users.keys())[0]
            if output_node.op != 'call_module':
                return find_sink(output_node)
            else:
                return output_node

        for sdpa_node in sdpa_nodes:
            value_input = sdpa_node.args[-1]

            value_node = find_src(value_input)
            output_node = find_sink(value_input)
            sink_module = get_module(graph_module, output_node.target)
            src_module = get_module(graph_module, value_node.target)
            sink_weight = get_weight_sink(sink_module)
            src_weight = get_weight_source(src_module)
            sink_eq_indexes = EqualizationIndexes(0, sink_weight.shape[0], 0)

            # TODO: restore fusing of Value/Output regions
            # src_eq_indexes = EqualizationIndexes(0, src_weight.shape[0], 0)

            region = Region(
                sinks={'sink_sdpa': sink_eq_indexes}, name_to_module={'sink_sdpa': sink_module})
            regions.append(region)

            for m in graph_module.modules():
                if isinstance(m, ScaledDotProductAttention):
                    m.pre_process_q = functional_rotate_input
                    m.pre_process_k = functional_rotate_input
        return regions

    def apply(self,
              graph_model: GraphModule) -> Union[Tuple[GraphModule, List[Transform]], GraphModule]:
        rewriters = []
        regions = _extract_regions(
            graph_model,
            state_impl_kwargs={
                'supported_srcs': self.supported_srcs,
                'supported_sinks': self.supported_sinks,
                'scale_invariant_layers': self.scale_invariant_layers,
                'scale_invariant_function': self.scale_invariant_function})
        expanded_regions = []
        self.find_module_by_name(graph_model, expanded_regions)
        eq_layers = set()
        orphan_regions = []

        if self.orphan_sink:
            blacklist_orphan_layers = self.blacklist_layers + self.layers_to_expand
            self.find_module(graph_model, orphan_regions, blacklist_layers=blacklist_orphan_layers)

        if len(expanded_regions) > 0:
            parameter_number_pre = 0
            for m in graph_model.parameters():
                parameter_number_pre += m.numel()
            logging.info(f"{len(expanded_regions)} layers will be expanded during rotation")

        if self.sdpa_regions:
            sdpa_regions = self.rotate_sdpa(graph_model)
            regions.extend(sdpa_regions)

        for r in regions:
            id_list = [id(r.name_to_module[sink_name]) for sink_name in r.sinks_names]
            eq_layers.update(id_list)

        # We check if any of the expanded region overlap with the fused regions.
        # If so, we need to apply expanded rotation after the fused one.
        # Furthremore, this is not compatible with optimized rotations.
        overlap = False
        for e_r in expanded_regions:
            # Layerwise have only a single sink named 'sinks0'
            id_sink = id(e_r.get_module_from_name('sinks0'))
            if id_sink in eq_layers:
                overlap = True

        if overlap:
            assert not self.use_parametrized_rotations, "Overlap between expanded and optimized region not supported"
            first_set, second_set = regions, expanded_regions
        else:
            first_set, second_set = expanded_regions, regions

        # We update mergeable regions to include also non-mergeable ones
        for o_r in orphan_regions:
            # Layerwise have only a single sink named 'sinks0'
            id_sink = id(o_r.get_module_from_name('sinks0'))
            if id_sink not in eq_layers:
                regions.append(o_r)

        if self.rotate_matmul:
            self.rotate_matmuls(graph_model)
        if len(regions) > 0:
            rewriters.extend(
                _apply_rotate(
                    graph_model,
                    first_set,
                    self.full_rotation_method,
                    fuse_rotations=not self.use_parametrized_rotations))
            rewriters.extend(
                _apply_rotate(
                    graph_model,
                    second_set,
                    self.full_rotation_method,
                    fuse_rotations=not self.use_parametrized_rotations))
            if len(expanded_regions) > 0:
                parameter_number_post = 0
                for m in graph_model.parameters():
                    parameter_number_post += m.numel()
                logging.info(
                    f"Added {parameter_number_post - parameter_number_pre} parameters to the model")

        if self.return_rewriters:
            return graph_model, rewriters
        else:
            return graph_model


class LayerNormToRMS(GraphTransform):

    def __init__(self, return_rewriters=False) -> None:
        super(LayerNormToRMS, self).__init__()
        self.supported_srcs = (nn.Linear, nn.Embedding)
        self.supported_sinks = (nn.LayerNorm)
        self.return_rewriters = return_rewriters
        assert RMSNorm is not object, 'Update your Pytorch version to 2.4+'

    def apply(self, graph_model: GraphModule) -> GraphModule:
        regions = _extract_regions(
            graph_model,
            state_impl_kwargs={
                'supported_srcs': self.supported_srcs, 'supported_sinks': self.supported_sinks})

        rewriters = []
        if len(regions) > 0:
            for region in regions:
                for src in region.srcs:
                    linear = region.get_module_from_name(src)
                    if isinstance(linear, torch.nn.Embedding):
                        dim = -1
                    else:
                        dim = -2
                    linear_dtype = linear.weight.data.dtype
                    W_ = linear.weight.data.double()
                    linear.weight.data = W_ - W_.mean(dim=dim, keepdim=True)
                    linear.weight.data = linear.weight.data.to(linear_dtype)
                    if hasattr(linear, 'bias') and linear.bias is not None:
                        b_ = linear.bias.data.double()
                        linear.bias.data = b_ - b_.mean()
                        linear.bias.data = linear.bias.data.to(linear_dtype)
                for sink in region.sinks:
                    layer_norm = region.get_module_from_name(sink)
                    del layer_norm.bias
                    layer_norm_dtype = layer_norm.weight.data.dtype
                    rewriters.append(
                        ModuleToModuleByInstance(layer_norm, RMSNorm, dtype=layer_norm_dtype))
            for r in rewriters:
                graph_model = r.apply(graph_model)
        if self.return_rewriters:
            return graph_model, rewriters
        else:
            return graph_model


class MergeLnAffine(GraphTransform):

    def __init__(self) -> None:
        super(MergeLnAffine, self).__init__()
        self.supported_srcs = (RMSNorm, nn.LayerNorm)
        self.supported_sinks = (nn.Linear)

    def apply(self, graph_model: GraphModule) -> GraphModule:
        regions = _extract_regions(
            graph_model,
            state_impl_kwargs={
                'supported_srcs': self.supported_srcs, 'supported_sinks': self.supported_sinks})

        if len(regions) > 0:
            scaled_biases = set()
            for region in regions:
                layernorm_module_name = next(iter(region.srcs))
                layernorm_module = region.get_module_from_name(layernorm_module_name)
                if not layernorm_module.elementwise_affine:
                    continue
                for name, indexes in region.sinks.items():
                    module = region.get_module_from_name(name)
                    scale_bias = id(module) not in scaled_biases
                    _merge_ln(layernorm_module, module, scale_bias_by_weight=scale_bias)

                    scaled_biases.add(id(module))
                layernorm_module.weight.data.fill_(1.)
                if hasattr(layernorm_module, 'bias'):
                    layernorm_module.bias.data.fill_(0.)
        return graph_model


class LayerwiseActivationRotation(RotationEqualization):

    def __init__(self, blacklist_layer=None, layers_to_expand=None):
        super().__init__(blacklist_layer, layers_to_expand)

        self.supported_sinks = (nn.Linear)

    def apply(self, model: nn.Module) -> nn.Module:

        blacklist_orphan_layers = self.blacklist_layers + self.layers_to_expand
        regions: List[Region] = []
        self.find_module(model, regions, blacklist_layers=blacklist_orphan_layers)
        expanded_regions = []
        self.find_module_by_name(model, expanded_regions)

        if len(expanded_regions) > 0:
            regions.extend(expanded_regions)
        if len(regions) > 0:
            _apply_rotate(model, regions)
        return model
