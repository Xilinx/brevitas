"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from accelerate.hooks import remove_hook_from_module
import torch
from tqdm import tqdm

from brevitas.graph.gptq import gptq_mode
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn
from brevitas_examples.optimum.utils import offload_model
from brevitas_examples.optimum.utils import remove_hooks


@torch.no_grad()
def gptq_iter(curr_layer, inps, outs, cached_values, act_order):
    curr_layer = curr_layer.cuda()
    with gptq_mode(curr_layer, use_quant_activations=False, act_order=act_order) as gptq:
        gptq_layer = gptq.model
        for _ in range(gptq.num_layers):
            for j in range(len(inps)):
                curr_inp = inps[j].unsqueeze(0).cuda()
                gptq_layer(curr_inp, **cached_values)
            gptq.update()
    for j in range(len(inps)):
        inp = inps[j].unsqueeze(0).cuda()
        curr_out = curr_layer(inp, **cached_values)[0]
        outs[j] = curr_out
    curr_layer.cpu()
    return outs


@torch.no_grad()
def apply_gptq(model, dataloader, act_order=True, block_name=None):
    model, _ = offload_model(model)
    with gptq_mode(model,
                   use_quant_activations=False,
                   act_order=act_order,
                   create_weight_orig=False) as gptq:
        gptq_model = gptq.model
        for _ in tqdm(range(gptq.num_layers)):
            for input_ids in dataloader:
                gptq_model(input_ids=input_ids.cuda())
            gptq.update()
    # Remove all accelerate hooks
    remove_hooks(model)
