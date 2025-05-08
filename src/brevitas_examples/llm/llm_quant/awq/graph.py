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
from brevitas.graph.equalize import _cross_layer_equalization
from brevitas.graph.equalize import Region
from brevitas.utils.parametrization_utils import ScaleWeightParametrization


# The Regions in AWQ are defined as subclasses of Region, for compatibility
# with Brevitas equalization logic
@dataclass(eq=True, frozen=True)
class RegionAWQ(Region):
    # Module for which to capture float/quantized outputs
    # in AWQ optimization
    block: Optional[nn.Module] = field(default=None)

    @property
    def prev_op(self) -> nn.Module:
        # The AWQ regions have a single previous operator
        return self.get_module_from_name(next(iter(self.srcs)))

    @property
    def repr_sink(self) -> nn.Module:
        # Retrives a representative sink for the region
        return self.get_module_from_name(next(iter(self.sinks)))


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

    def __init__(self) -> None:
        super(EqualizeAWQ, self).__init__()
        # These attributes are only kept to reuse the _cross_layer_equalization method
        self._merge_bias = True
        self._bias_shrinkage = 'vaiq'
        self._scale_computation_type = 'maxabs'

    def _apply_scaling_rewriters(self, model: Union[GraphModule, nn.Module],
                                 region: Region) -> List[Transform]:
        _, rewriters = _cross_layer_equalization(
            model,
            region,
            merge_bias=self._merge_bias,
            bias_shrinkage=self._bias_shrinkage,
            scale_computation_type=self._scale_computation_type,
            fuse_scaling=False)
        # Scaling factors are set to identity
        for r in rewriters:
            if isinstance(r, ModuleInstanceRegisterParametrization):
                if isinstance(r.transform_module, ScaleWeightParametrization):
                    r.transform_module.scaling_factor.data = torch.ones_like(
                        r.transform_module.scaling_factor.data)
        return rewriters

    def apply(self,
              model: Union[GraphModule, nn.Module],
              regions: List[Region] = None) -> List[Transform]:
        # Only apply scaling parametrization in non-orphan regions
        regions = [region for region in regions if len(region.srcs) > 0]
        rewriters = []
        for region in regions:
            rewriters.extend(self._apply_scaling_rewriters(model, region))
        return model, regions, rewriters
