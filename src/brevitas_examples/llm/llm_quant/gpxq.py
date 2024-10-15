# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import torch
from tqdm import tqdm

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gpfq import GPFQv2
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode

from .axe import A2GPFQ
from .axe import A2GPTQ


@torch.no_grad()
def apply_gptq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        use_quant_activations=True,
        create_weight_orig=False,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=128):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gptq_class = partial(
            A2GPTQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gptq_class = GPTQ
    with gptq_mode(model,
                   act_order=act_order,
                   group_of_parallel_layers=group_of_parallel_layers,
                   use_quant_activations=use_quant_activations,
                   create_weight_orig=create_weight_orig,
                   gptq_class=gptq_class) as gptq:
        gptq_model = gptq.model
        for _ in tqdm(range(gptq.num_layers)):
            for inps in dataloader:
                gptq_model(**inps)
            gptq.update()


@torch.no_grad()
def apply_gpfq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=128):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gpfq_class = partial(
            A2GPFQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gpfq_class = GPFQv2
    with gpfq_mode(model,
                   act_order=act_order,
                   group_of_parallel_layers=group_of_parallel_layers,
                   gpfq_class=gpfq_class) as gpfq:
        gpfq_model = gpfq.model
        for _ in tqdm(range(gpfq.num_layers)):
            for inps in dataloader:
                gpfq_model(**inps)
            gpfq.update()
