"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch

from brevitas.graph.gptq import gptq_mode
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn


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
def apply_gptq(model, dataloader, nsamples, act_order=True, seqlen=2048):
    apply_layer_ptq_fn(
        model, dataloader, nsamples, inference_fn=gptq_iter, seqlen=seqlen, act_order=act_order)
