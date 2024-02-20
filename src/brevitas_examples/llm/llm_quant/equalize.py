"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizeGraph


@torch.no_grad()
def apply_act_equalization(model, act_equalization_type, dataloader, alpha=0.5):
    if act_equalization_type == 'layerwise':
        with activation_equalization_mode(model, alpha, add_mul_node=True, layerwise=True):
            for inps in tqdm(dataloader):
                model(**inps)

    elif act_equalization_type == 'fx':
        assert model is not None, "FX Model is required to perform FX SmoothQuant"
        with activation_equalization_mode(model,
                                          alpha,
                                          add_mul_node=False,
                                          layerwise=False,
                                          co_optimize_act_weights=True):
            for inps in tqdm(dataloader):
                model(**inps)

    else:
        raise RuntimeError(f"{act_equalization_type} not supported.")


@torch.no_grad()
def apply_weight_equalization(graph_model, scale_computation_type='range'):
    EqualizeGraph(scale_computation_type=scale_computation_type).apply(graph_model)
