"""
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode


@torch.no_grad()
def apply_gptq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        use_quant_activations=True,
        create_weight_orig=False):
    with gptq_mode(model,
                   act_order=act_order,
                   group_of_parallel_layers=group_of_parallel_layers,
                   use_quant_activations=use_quant_activations,
                   create_weight_orig=create_weight_orig) as gptq:
        gptq_model = gptq.model
        for _ in tqdm(range(gptq.num_layers)):
            for inps in dataloader:
                gptq_model(**inps)
            gptq.update()


@torch.no_grad()
def apply_gpfq(model, dataloader, act_order=True, group_of_parallel_layers=None):
    with gpfq_mode(model, act_order=act_order,
                   group_of_parallel_layers=group_of_parallel_layers) as gpfq:
        gpfq_model = gpfq.model
        for _ in tqdm(range(gpfq.num_layers)):
            for inps in dataloader:
                gpfq_model(**inps)
            gpfq.update()
