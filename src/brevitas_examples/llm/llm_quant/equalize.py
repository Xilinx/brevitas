"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from accelerate.hooks import remove_hook_from_module
import torch
from tqdm import tqdm

from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizeGraph
from brevitas_examples.optimum.utils import offload_model
from brevitas_examples.optimum.utils import remove_hooks


@torch.no_grad()
def activation_equalization_iter(curr_layer, inps, outs, cached_values, alpha):
    curr_layer = curr_layer.cuda()
    with activation_equalization_mode(curr_layer, alpha, add_mul_node=True, layerwise=True):
        for j in range(len(inps)):
            inp = inps[j].unsqueeze(0).cuda()
            curr_out = curr_layer(inp, **cached_values)[0]
            outs[j] = curr_out
    curr_layer.cpu()
    return outs


@torch.no_grad()
def apply_act_equalization(model, act_equalization_type, dataloader, alpha=0.5):
    model = offload_model(model)
    if act_equalization_type == 'layerwise':
        with activation_equalization_mode(model, alpha, add_mul_node=True, layerwise=True):
            for inps in tqdm(dataloader):
                inps = {k:v.cuda() for (k,v) in inps.items()}
                model(**inps)
    elif act_equalization_type == 'fx':
        assert model is not None, "FX Model is required to perform FX SmoothQuant"
        with activation_equalization_mode(model,
                                          alpha,
                                          add_mul_node=False,
                                          layerwise=False,
                                          co_optimize_act_weights=True):
            for inps in tqdm(dataloader):
                inps = {k:v.cuda() for (k,v) in inps.items()}
                model(**inps)

    else:
        raise RuntimeError(f"{act_equalization_type} not supported.")
    # Remove all accelerate hooks
    remove_hooks(model)


@torch.no_grad()
def apply_weight_equalization(graph_model, scale_computation_type='range'):
    EqualizeGraph(scale_computation_type=scale_computation_type).apply(graph_model)
