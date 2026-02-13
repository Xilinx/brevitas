# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import operator
from typing import List
from typing import Optional
import warnings

import torch
from torch.fx import GraphModule
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _channel_maxabs
from brevitas.graph.equalize import _scale_invariant_layers
from brevitas.graph.equalize import find_srcs
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import Region
from brevitas.graph.equalize import RegionWalkMixin
from brevitas.graph.equalize import WalkRegionState
from brevitas.graph.utils import find_node_for_module
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.utils.logging import setup_logger

logging = setup_logger(__name__)

__all__ = ['GraphPermutationEqualization', 'rotate_permute_mode']

# Initialize permutation-invariant layers from scale-invariant layers
_permute_invariant_layers = list(_scale_invariant_layers)
_permute_invariant_layers.extend([torch.nn.GELU, torch.nn.SELU])

# Try to add HuggingFace activations
try:
    from transformers.activations import ACT2CLS
    activations = [x if not isinstance(x, tuple) else x[0] for x in ACT2CLS.values()]
    _permute_invariant_layers.extend(activations)
except:
    pass

# Try to add RMSNorm
try:
    from torch.nn import RMSNorm
    _permute_invariant_layers.append(RMSNorm)
except:
    pass

_permute_invariant_layers = tuple(_permute_invariant_layers)
_permute_invariant_functions = (torch.nn.functional.silu,)

# Dictionary to store registered permutation methods
_PERMUTATION_METHODS = {}


def register_permutation_method(name: str):
    """Register a permutation method for block rotations.

    Args:
        name: The name of the permutation method (e.g., "zigzag", "massdiff")

    Examples:
        >>> @register_permutation_method("my_permute")
        ... def my_permute_method(x, block_rotation_dim):
        ...     return torch.arange(x.shape[-1])
    """

    def _wrapper(permute_fn):
        if name in _PERMUTATION_METHODS:
            logging.warning(
                "The permutation method '%s' already exists and will be "
                "overwritten by %s.",
                name,
                permute_fn.__name__,
            )
        _PERMUTATION_METHODS[name] = permute_fn
        return permute_fn

    return _wrapper


def get_permutation_method(name: str):
    """Get a registered permutation method by name."""
    if name not in _PERMUTATION_METHODS:
        available = list(_PERMUTATION_METHODS.keys())
        raise ValueError(
            f"Permutation method '{name}' not found. "
            f"Available methods: {available}")
    return _PERMUTATION_METHODS[name]


@register_permutation_method("zigzag")
def zigzag_permute(x, block_rotation_dim):
    if x.shape[-1] == block_rotation_dim:
        return torch.arange(block_rotation_dim).to(x.device)
    scores = _channel_maxabs(x, dim=0)
    _, indexes = torch.sort(scores, descending=True)
    # Inline zigzag sort logic
    indexes = indexes.view(block_rotation_dim, indexes.shape[-1] // block_rotation_dim)
    indexes[1::2] = torch.flip(indexes[1::2], dims=[1])
    indexes = indexes.t()
    indexes = indexes.flatten()
    return indexes


@register_permutation_method("random")
def random_permute(x, block_rotation_dim):
    if x.shape[-1] == block_rotation_dim:
        return torch.arange(block_rotation_dim).to(x.device)
    indexes = torch.randperm(x.shape[-1]).to(x.device)
    return indexes


@register_permutation_method("absmax")
def absmax_permute(x, block_rotation_dim):
    if x.shape[-1] == block_rotation_dim:
        return torch.arange(block_rotation_dim).to(x.device)
    scores = _channel_maxabs(x, dim=0)
    _, indexes = torch.sort(scores, descending=True)
    return indexes


@register_permutation_method("massdiff")
def massdiff_permute(x, block_rotation_dim):
    if x.shape[-1] == block_rotation_dim:
        return torch.arange(block_rotation_dim).to(x.device)
    # initialize the blocks based on absmax scores
    scores = torch.abs(x).mean(dim=0)
    _, indexes = torch.sort(scores, descending=True)
    num_blocks = x.shape[-1] // block_rotation_dim
    # initialize the block norms and indexes
    block_norm = torch.stack([torch.abs(x[:, i]) for i in indexes[:num_blocks]], dim=1)
    block_idxs = [[i] for i in indexes[:num_blocks]]
    for i in indexes[num_blocks:]:
        # find the block that will have the minimum l1-norm after adding the new index
        norms_after_adding = block_norm + torch.abs(x[:, i]).unsqueeze(1)
        norms_after_adding = torch.mean(norms_after_adding, dim=0)
        min_block = torch.argmin(norms_after_adding)
        # update the block norm and indexes
        block_norm[:, min_block] += torch.abs(x[:, i])
        block_idxs[min_block].append(i)
        # mark block as full
        if (len(block_idxs[min_block]) == block_rotation_dim):
            block_norm[:, min_block] = float('inf')
    indexes = torch.tensor(block_idxs).flatten()
    return indexes


class rotate_permute_mode:

    def __init__(
            self,
            model: GraphModule,
            permute_fn: str = 'massdiff',
            block_rotation_dim: Optional[int] = None,
            **kwargs):
        # rotate_permute_mode performs a specific transformation sequence:
        # 1. Identify regions → 2. Apply permutations → 3. Apply rotations
        # This sequencing requires delay_rewriters=True (defer rotation application)
        # and return_rewriters=True (to access rotations for later application)
        if not kwargs.pop('delay_rewriters', True) or not kwargs.pop('return_rewriters', True):
            warnings.warn(
                "delay_rewriters and return_rewriters must be True for rotate_permute_mode, ",
                "overwriting provided value(s).")
        kwargs['delay_rewriters'] = True
        kwargs['return_rewriters'] = True

        self.rotation = GraphRotationEqualization(block_rotation_dim=block_rotation_dim, **kwargs)
        self.permutation = GraphPermutationEqualization(
            block_rotation_dim=block_rotation_dim, permute_fn=permute_fn)
        self.model = model
        self.rewriters = None

    def __enter__(self):
        model, rewriters = self.rotation.apply(self.model)
        self.model = model
        self.rewriters = rewriters

        # NOTE: permutations are tied to block rotations here
        permute_regions = self.rotation.get_regions()
        self.model = self.permutation.setup(
            self.model,
            permute_regions,
            disable_for_fused=self.rotation.disable_block_rotation_for_fused)
        return self

    def __exit__(self, *args, **kwargs):
        self.model = self.permutation.apply(self.model)
        self.permutation.cleanup()


class GraphPermutationEqualization(GraphTransform, RegionWalkMixin):
    """
    A class for managing and applying permutations to a computational graph.

    This class is designed to analyze and modify computational graphs by identifying
    regions of interest, collecting statistics, and applying permutations to optimize
    or modify the graph's behavior. It supports various neural network layers and
    operations, and provides hooks for collecting forward pass statistics.
    """

    def __init__(self, block_rotation_dim: int, permute_fn: str = 'massdiff'):
        # Initialize RegionWalkMixin
        mul_ops = [torch.mul, operator.mul, operator.imul, operator.__mul__, operator.__imul__]
        residual_fns = [torch.add, operator.add, operator.iadd, operator.__add__, operator.__iadd__]
        residual_fns.extend(mul_ops)

        base_state_kwargs = {
            'supported_srcs': (nn.Embedding, RotatedModule, nn.Linear),
            'supported_sinks': (nn.Linear, RotatedModule),
            'scale_invariant_layers': _permute_invariant_layers,
            'scale_invariant_functions': _permute_invariant_functions,
            'residual_fns': tuple(residual_fns),}
        RegionWalkMixin.__init__(self, **base_state_kwargs)

        # Initialize other attributes
        self.hooks = []
        self.hooked_modules = set()
        self.regions = list()
        self.float_act_map = dict()
        self.float_act_dev = dict()
        self.block_rotation_dim = block_rotation_dim
        self.permute_fn = get_permutation_method(permute_fn)

    def setup(
            self,
            graph_model: GraphModule,
            regions: List[Region],
            disable_for_fused: bool = False) -> GraphModule:
        """
        Setup phase: filter regions, extract, and install hooks.
        """
        # Filter regions for permutation
        filtered_regions = self._filter_regions(regions, disable_for_fused)

        # Extract permute regions from filtered regions
        self._extract_regions(graph_model, filtered_regions)

        # Setup forward hooks
        self._setup_hooks()

        return graph_model

    def forward_stats_hook(self, module, *args, name, batch_dim=0, **kwargs):
        # Check for MHA Cross attention, and if found, skip it
        # When using hf/accelerate, we need to check the signature of the original forward
        forward_to_check = module._old_forward if hasattr(
            module, '_old_forward') else module.forward
        kwargs.update(zip(forward_to_check.__code__.co_varnames[1:], args[:-1]))
        if 'query' in kwargs and 'key' in kwargs and 'value' in kwargs:
            if kwargs['query'].data_ptr() != kwargs['key'].data_ptr() != kwargs['value'].data_ptr():
                return

        INPUT_NAMES = ('input', 'inp', 'query', 'hidden_states', 'x')
        inp_kwarg = [x for x in kwargs.keys() if x in INPUT_NAMES][0]
        inp = kwargs[inp_kwarg][0]

        # Extra check for batch_dim
        if hasattr(inp, 'names') and 'N' in inp.names:
            batch_dim = inp.names.index('N')
            inp.rename_(None)
            inp = inp.transpose(0, batch_dim)

        inp = inp.reshape(-1, inp.shape[-1])  # [batch_size * seq_len, dim]
        if name not in self.float_act_map:
            self.float_act_map[name] = []
            self.float_act_dev[name] = inp.device
        self.float_act_map[name].append(inp.detach().cpu())

    def _setup_hooks(self):
        for region in self.regions:
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

            for name in region.sinks_names:
                module = region.get_module_from_name(name)
                if module not in self.hooked_modules:
                    self.hooked_modules.add(module)
                    hook_fn = partial(self.forward_stats_hook, name=name, batch_dim=batch_dim)
                    h = module.register_forward_hook(hook_fn)
                    self.hooks.append(h)

    def _filter_regions(self, regions: list, disable_for_fused: bool = False) -> List[Region]:
        """
        Filter regions to identify which should have permutations applied.
        """
        permute_regions = list()
        for region in regions:
            # Permutations are only applied to regions that use block rotations
            apply_block_rotation = self.block_rotation_dim is not None
            # Optionally disable permutations for fused rotations
            if disable_for_fused and (len(region.srcs) > 0):
                apply_block_rotation = False
            if apply_block_rotation:
                # Check if block rotation is compatible with the current shape
                if (region.max_shape_sinks // self.block_rotation_dim > 1) and \
                    (region.max_shape_sinks % self.block_rotation_dim == 0):
                    permute_regions.append(region)
        return permute_regions

    def _extract_regions(self, graph_model, regions):
        """
        Extract and process permutation regions from the graph model.
        """
        for region in regions:
            # Directly add regions that already have sources identified
            if (len(region.srcs) > 0):
                # Skip the SDPA regions; potential head alignment issues
                if 'value_sdpa' not in region.srcs_names:
                    self.regions.append(region)
                continue

            # Create a new state for the online region
            state = WalkRegionState(**self.full_state_kwargs)

            # Add all sinks from the region to the state
            for sink_name, sink_wrapper in region.sinks.items():
                module = region.get_module_from_name(sink_name)
                node = find_node_for_module(graph_model, module)
                assert node is not None, f"Error: node {module} not found in graph"
                eq_indexes = sink_wrapper.equalization_indexes
                state.add_sinks(node.target, module, eq_indexes)
                find_srcs(graph_model, node, state)

            # Create a new region with updated sources but same sinks
            new_region = Region.from_dicts(
                srcs=state.srcs,
                sinks=state.sinks,
                name_to_module=state.name_to_module,
                expand_region=region.expand_region)
            self.regions.append(new_region)

    @staticmethod
    def permute_region(region, list_of_act_val, block_rotation_dim, permute_fn, device):
        """
        Apply permutation to a region by calculating permutation indexes and updating
        the source and sink weights accordingly.
        """
        # If equalization criteria are not met, return without doing anything
        if not region.is_valid_activation_equalization:
            return

        list_of_act_val_shapes = [act_val.shape for act_val in list_of_act_val]
        if len(list_of_act_val_shapes) > 0:
            shape_0 = list_of_act_val_shapes[0]
            if any(shape_0 != shape for shape in list_of_act_val_shapes):
                return

        list_of_act_val = torch.cat(list_of_act_val, dim=0).to(device)
        new_indexes = permute_fn(list_of_act_val, block_rotation_dim=block_rotation_dim)

        for src in region.srcs.values():
            src.permute(new_indexes)
        for sink in region.sinks.values():
            sink.permute(new_indexes)

    def apply(self, graph_model: GraphModule) -> GraphModule:
        """
        Apply permutations to the graph model.
        """
        for region in tqdm(self.regions, "Calculating permutations..."):
            # Collect all activation values for this region
            list_of_act_val = []
            for name in region.sinks_names:
                act_vals = self.float_act_map.pop(name)
                if act_vals is None or len(act_vals) == 0:
                    continue
                list_of_act_val.extend(act_vals)
            # Calculate permutation and apply to this region
            self.permute_region(
                region,
                list_of_act_val=list_of_act_val,
                block_rotation_dim=self.block_rotation_dim,
                permute_fn=self.permute_fn,
                device=self.float_act_dev[region.sinks_names[0]])
        return graph_model

    def cleanup(self):
        for h in self.hooks:
            h.remove()
