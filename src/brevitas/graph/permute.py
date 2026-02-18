# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import operator
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
from torch.fx import GraphModule
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _channel_maxabs
from brevitas.graph.equalize import _scale_invariant_layers
from brevitas.graph.equalize import _UNSUPPORTED_OP
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
_permute_invariant_layers.extend([torch.nn.GELU, torch.nn.SELU, torch.nn.SiLU])

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
    """
    Register a permutation method.

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
def zigzag_permute(x, block_size):
    if x.shape[-1] == block_size:
        return torch.arange(block_size).to(x.device)
    scores = _channel_maxabs(x, dim=0)
    _, indexes = torch.sort(scores, descending=True)
    # Inline zigzag sort logic
    indexes = indexes.view(block_size, indexes.shape[-1] // block_size)
    indexes[1::2] = torch.flip(indexes[1::2], dims=[1])
    indexes = indexes.t()
    indexes = indexes.flatten()
    return indexes


@register_permutation_method("random")
def random_permute(x, block_size):
    if x.shape[-1] == block_size:
        return torch.arange(block_size).to(x.device)
    indexes = torch.randperm(x.shape[-1]).to(x.device)
    return indexes


@register_permutation_method("absmax")
def absmax_permute(x, block_size):
    if x.shape[-1] == block_size:
        return torch.arange(block_size).to(x.device)
    scores = _channel_maxabs(x, dim=0)
    _, indexes = torch.sort(scores, descending=True)
    return indexes


@register_permutation_method("massdiff")
def massdiff_permute(x, block_size):
    if x.shape[-1] == block_size:
        return torch.arange(block_size).to(x.device)
    # initialize the blocks based on absmax scores
    scores = torch.abs(x).mean(dim=0)
    _, indexes = torch.sort(scores, descending=True)
    num_blocks = x.shape[-1] // block_size
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
        if (len(block_idxs[min_block]) == block_size):
            block_norm[:, min_block] = float('inf')
    indexes = torch.tensor(block_idxs).flatten()
    return indexes


class GraphPermutationEqualization(GraphTransform, RegionWalkMixin):
    """
    A class for managing and applying permutations to a computational graph
    """

    def __init__(
            self,
            block_size: int,
            permute_fn: str = 'massdiff',
            extra_state_kwargs: Optional[Dict[str, Tuple[Type[nn.Module]]]] = None):
        assert isinstance(block_size, int) and block_size > 1, "Error: expected an integer > 1."
        assert permute_fn in _PERMUTATION_METHODS, f"Error: {permute_fn} is not registered."

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
        RegionWalkMixin.__init__(self, **base_state_kwargs, extra_state_kwargs=extra_state_kwargs)

        # Initialize other attributes
        self.hooks = []
        self.hooked_modules = set()
        self.regions = list()
        self.float_act_map = dict()
        self.float_act_dev = dict()
        self.block_size = block_size
        self.permute_fn = get_permutation_method(permute_fn)

    def setup(self, graph_model: GraphModule, regions: List[Region]) -> GraphModule:
        """Extract regions and setup hooks"""
        self._extract_regions(graph_model, regions)
        self._setup_hooks()
        return graph_model

    def forward_stats_hook(self, module, *args, name, batch_dim=0, **kwargs):
        inp, batch_dim = self._process_input(module, args, kwargs, batch_dim, use_inp=True)

        if inp is None:
            return

        if hasattr(inp, 'names') and 'N' in inp.names:
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

    def _is_compatible_region(self, region: Region) -> bool:
        if (region.max_shape_sinks // self.block_size > 1) and \
            (region.max_shape_sinks % self.block_size == 0):
            return True
        return False

    def _extract_regions(self, graph_model, regions):
        """
        Extract and process permutation regions from the graph model.
        """
        for region in regions:
            # Check if block size is compatible with the current shape
            if not self._is_compatible_region(region):
                continue

            # Directly add regions that already have sources identified
            if (len(region.srcs) > 0):
                # Skip the SDPA regions; potential head alignment issues
                if 'value_sdpa' not in region.srcs_names:
                    self.regions.append(region)
                continue

            # Skip if equalization criteria are not met
            if not region.is_valid_activation_equalization:
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

            # Skip region creation if unsupported operations were encountered
            if _UNSUPPORTED_OP in state.sinks:
                continue

            # Create a new region with updated sources but same sinks
            new_region = Region.from_dicts(
                srcs=state.srcs,
                sinks=state.sinks,
                name_to_module=state.name_to_module,
                expand_region=region.expand_region)
            self.regions.append(new_region)

    @staticmethod
    def permute_region(region, list_of_act_val, block_size, permute_fn, device):
        """
        Apply permutation to a region by calculating permutation indexes and updating
        the source and sink weights accordingly.
        """
        list_of_act_val_shapes = [act_val.shape for act_val in list_of_act_val]
        if len(list_of_act_val_shapes) > 0:
            shape_0 = list_of_act_val_shapes[0]
            if any(shape_0 != shape for shape in list_of_act_val_shapes):
                return

        list_of_act_val = torch.cat(list_of_act_val, dim=0).to(device)
        new_indexes = permute_fn(list_of_act_val, block_size=block_size)

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
                block_size=self.block_size,
                permute_fn=self.permute_fn,
                device=self.float_act_dev[region.sinks_names[0]])
        return graph_model

    def cleanup(self):
        for h in self.hooks:
            h.remove()


class rotate_permute_mode:
    """
    Context manager for applying rotation and permutation equalization.

    Args:
        model: The graph module to transform
        rotation: Pre-initialized GraphRotationEqualization instance
        permute_fn: Permutation method name
        block_size: Block size for permutations
        disable_for_fused_rotations: Whether to disable permutations for fused rotations
    """

    def __init__(
            self,
            model: GraphModule,
            rotation: GraphRotationEqualization,
            block_size: int,
            permute_fn: str = 'massdiff',
            disable_for_fused_rotations: bool = False,
            extra_state_kwargs: Optional[Dict[str, Tuple[Type[nn.Module]]]] = None):

        assert rotation is not None and isinstance(rotation, GraphRotationEqualization), \
            "Error: expected GraphRotationEqualization instance"
        assert rotation.delay_rewriters, "Error: expected rotation.delay_rewriters=True"
        assert rotation.return_rewriters, "Error: expected rotation.return_rewriters=True"
        assert isinstance(block_size, int) and block_size > 1, "Error: expected integer > 1"

        self.model = model
        self.rotation = rotation
        self.permute_fn = permute_fn
        self.block_size = block_size
        self.disable_for_fused_rotations = disable_for_fused_rotations

        self.permutation = GraphPermutationEqualization(
            block_size=block_size, permute_fn=permute_fn, extra_state_kwargs=extra_state_kwargs)
        self.rewriters = []

    def _filter_regions(self, regions: List[Region]) -> List[Region]:
        """
        Given rotation regions, filter out regions where permutations shouldn't be applied
        """
        permute_regions = []
        for region in regions:
            # Optionally disable permutations for fused rotations by skipping those regions
            if self.disable_for_fused_rotations and (len(region.srcs) > 0):
                continue
            permute_regions.append(region)
        return permute_regions

    def __enter__(self):
        # Apply rotations and get rewriters
        model, rewriters = self.rotation.apply(self.model)
        self.model = model
        self.rewriters = rewriters

        # Filter and setup permutation hooks based on rotation regions
        regions = self.rotation.get_regions()
        regions = self._filter_regions(regions)
        self.model = self.permutation.setup(self.model, regions)
        return self

    def __exit__(self, *args, **kwargs):
        # Apply permutations and cleanup
        self.model = self.permutation.apply(self.model)
        self.permutation.cleanup()
