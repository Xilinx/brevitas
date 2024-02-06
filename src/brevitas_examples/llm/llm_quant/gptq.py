"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.gptq import gptq_mode


@torch.no_grad()
def apply_gptq(model, dataloader, act_order=True, group_of_parallel_layers=None):
    with gptq_mode(model,
                   use_quant_activations=False,
                   group_of_parallel_layers=group_of_parallel_layers,
                   act_order=act_order,
                   create_weight_orig=False) as gptq:
        gptq_model = gptq.model
        for _ in tqdm(range(gptq.num_layers)):
            for inps in dataloader:
                gptq_model(**inps)
            gptq.update()
