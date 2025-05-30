# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F

from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import ScaledDotProductAttention


def replace_sdpa_with_quantizable_layers(graph_model):
    fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
    graph_model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(graph_model)
    return graph_model


@torch.no_grad()
def add_zero_bias_to_linear(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear:
            if module.bias is None:
                module.register_parameter(
                    "bias",
                    torch.nn.Parameter(
                        torch.zeros((module.weight.shape[0],),
                                    device=module.weight.device,
                                    dtype=module.weight.dtype)),
                )
    return model
