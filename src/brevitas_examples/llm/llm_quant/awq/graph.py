# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional, Union

import torch
from torch import nn

from brevitas.graph.base import GraphModule
from brevitas.graph.base import GraphTransform
from brevitas.graph.base import ModuleInstanceRegisterParametrization
from brevitas.graph.base import Transform
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _cross_layer_equalization
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import extract_sdpa_regions
from brevitas.graph.equalize import Region
from brevitas.utils.parametrization_utils import ScaleWeightParametrization
from brevitas.utils.python_utils import longest_common_prefix
from brevitas.utils.python_utils import recurse_getattr


@dataclass(eq=True, frozen=True)
class RegionAWQ(Region):
    block: Optional[nn.Module] = field(default=None)
    use_kwargs: bool = field(default=False)

    @property
    def prev_op(self) -> nn.Module:
        return self.get_module_from_name(next(iter(self.srcs)))

    @property
    def repr_sink(self) -> nn.Module:
        return self.get_module_from_name(next(iter(self.sinks)))


def _find_common_ancestor_sinks(model: nn.Module, region: Region) -> nn.Module:
    module_name = longest_common_prefix(region.sinks_names).strip(".")
    return recurse_getattr(model, module_name)


def initialize_awq_region(model: nn.Module, region: Region) -> RegionAWQ:
    return RegionAWQ(
        srcs=region.srcs,
        sinks=region.sinks,
        acts=region.acts,
        name_to_module=region.name_to_module,
        block=_find_common_ancestor_sinks(model, region),
        use_kwargs=len(region.sinks) > 1,
    )


def extract_sinks_scaling_factor(sinks: List[nn.Module]) -> nn.Parameter:
    trainable_params = []
    # IDs of the scaling factors are tracked, as several modules can share
    # the same scaling
    ids_params = set()
    for sink in sinks:
        for module in sink.modules():
            # Sinks have their scaling factors set to True
            if isinstance(module, ScaleWeightParametrization) and module.use_inverse_scaling:
                if id(module.scaling_factor) not in ids_params:
                    ids_params.add(id(module.scaling_factor))
                    trainable_params.append(module.scaling_factor)
    # Verify that a single factor was retrieved
    assert len(trainable_params) == 1
    # Return the unique scaling factor for the set of sinks
    return trainable_params[0]


class EqualizeAWQ(GraphTransform):

    def __init__(
            self, sdpa_regions: bool = True, add_parametrizations_inplace: bool = False) -> None:
        super(EqualizeAWQ, self).__init__()
        self.sdpa_regions = sdpa_regions
        self.add_parametrizations_inplace = add_parametrizations_inplace
        self.blacklist_layers = ["lm_head", "embedding"]
        # These attributes are only kept to reuse the _cross_layer_equalization method
        self._merge_bias = True
        self._bias_shrinkage = 'vaiq'
        self._scale_computation_type = 'maxabs'

    def _extract_awq_regions(self, graph_model: GraphModule) -> List[Region]:
        regions = []
        # Add Value/Output region
        if self.sdpa_regions:
            sdpa_regions = extract_sdpa_regions(graph_model)
            regions.extend(sdpa_regions)
        # It is not possible to equalize through LayerNorm/BatchNorm as sink
        supported_sinks = tuple([
            x for x in _supported_layers if x not in (nn.LayerNorm, *_batch_norm)])
        regions.extend(
            _extract_regions(graph_model, state_impl_kwargs={'supported_sinks': supported_sinks}))
        # Remove regions containing blacklisted layers
        regions = [
            region for region in regions if not any(
                any(blacklist_layer in m_name
                    for m_name in region.name_to_module)
                for blacklist_layer in self.blacklist_layers)]
        return regions

    def _retrieve_scaling_rewriters(self, model: Union[GraphModule, nn.Module],
                                    region: Region) -> List[Transform]:
        _, rewriters = _cross_layer_equalization(
            model,
            region,
            merge_bias=self._merge_bias,
            bias_shrinkage=self._bias_shrinkage,
            scale_computation_type=self._scale_computation_type,
            fuse_scaling=False,
            parametrize_inplace=False)
        # Scaling factors are set to identity
        for r in rewriters:
            if isinstance(r, ModuleInstanceRegisterParametrization):
                if isinstance(r.transform_module, ScaleWeightParametrization):
                    r.transform_module.scaling_factor.data = torch.ones_like(
                        r.transform_module.scaling_factor.data)
        return rewriters

    def apply(self,
              model: Union[GraphModule, nn.Module],
              regions: Optional[List[Region]] = None) -> List[Transform]:
        # Try to identify regions if not passed directly
        if regions is None:
            regions = self._extract_awq_regions(model)
        else:
            regions = [region for region in regions if len(region.srcs) > 0]
        rewriters = []
        for region in regions:
            rewriters.extend(self._retrieve_scaling_rewriters(model, region))
        if self.add_parametrizations_inplace:
            for r in rewriters:
                model = r.apply(model)
        return model, regions, rewriters
