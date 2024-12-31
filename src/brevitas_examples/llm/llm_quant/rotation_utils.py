# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import List

from torch import nn
from torch.fx import GraphModule
import torch.nn.utils.parametrize as parametrize

from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import Transform
from brevitas.graph.equalize import EqualizationIndexes
from brevitas.graph.equalize import Region
from brevitas.graph.equalize import WalkRegionState
from brevitas.nn.equalized_layer import RotationWeightParametrization


def find_self_attention_rotation_regions(
        graph_model: GraphModule, head_dim: int, state_impl_kwargs=None) -> List[Region]:
    regions = []
    # See R2 rotation matrices in https://arxiv.org/pdf/2405.16406.
    for src_name, src_module in graph_model.named_modules():
        if "attn_v_proj" in src_name:
            if state_impl_kwargs is not None:
                state = WalkRegionState(**state_impl_kwargs)
            else:
                state = WalkRegionState()

            block_number_matches_src = re.findall(r'\d+', src_name)
            assert len(block_number_matches_src) == 2, "Could not identify block"
            block_number_src = int(block_number_matches_src[1])

            eq_indexes = EqualizationIndexes(0, head_dim, 0)
            state.add_srcs(src_name, src_module, eq_indexes)

            # Now the corresponding sink
            for sink_name, sink_module in graph_model.named_modules():
                if "attn_o_proj" in sink_name:
                    block_number_matches_sink = re.findall(r'\d+', sink_name)
                    assert len(block_number_matches_sink) == 2, "Could not identify block"
                    block_number_sink = int(block_number_matches_sink[1])
                    # If the blocks match, the region was identified
                    if block_number_src == block_number_sink:
                        eq_indexes = EqualizationIndexes(0, head_dim, state.offset)
                        state.add_sinks(sink_name, sink_module, eq_indexes)
            region = Region(
                srcs=dict(sorted(state.srcs.items())),
                sinks=dict(sorted(state.sinks.items())),
                name_to_module=state.name_to_module,
            )
            if region not in regions:
                regions.append(region)

    return regions


def fuse_rotations(model: nn.Module) -> None:
    for module in model.modules():
        # Check if the module has any parametrizations
        if hasattr(module, "parametrizations"):
            # Remove weight parametrizations
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            # We need to check again, in case the weight parametrizations were the only ones
            if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "bias"):
                parametrize.remove_parametrizations(module, "bias", leave_parametrized=True)


# TODO: Remove? We rely on ModuleInstanceRegisterParametrization
def extract_rewriters_unfused_rotations(model: nn.Module,
                                        rewriters: List[Transform]) -> List[Transform]:
    extra_rewriters = []
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            # Verify that the current module does not have already associated a RotatedModule
            if len([r for r in rewriters if r.old_module_instance is module and
                    isinstance(r, ModuleInstanceToModuleInstance)]) == 0:
                # Identity rewriter, only useful externaly
                rewriter = ModuleInstanceToModuleInstance(module, module)
                extra_rewriters.append(rewriter)
    return extra_rewriters


def extract_trainable_rotation_matrices(model: nn.Module) -> List[nn.Parameter]:
    trainable_rotations = []
    # We need to keep track of the IDs of the rotation matrices, as several modules
    # can share the same parametrized rotation.
    ids_rot = set()
    for module in model.modules():
        if isinstance(module, RotationWeightParametrization):
            if id(module.rot_mat) not in ids_rot:
                ids_rot.add(id(module.rot_mat))
                trainable_rotations.append(module.rot_mat)
    return trainable_rotations
