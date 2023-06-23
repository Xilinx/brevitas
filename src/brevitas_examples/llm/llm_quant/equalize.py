"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch

from brevitas.fx.brevitas_tracer import value_trace
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizeGraph
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn


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
def apply_act_equalization(model, dataloader, nsamples, seqlen=2048, alpha=0.5):
    apply_layer_ptq_fn(
        model,
        dataloader,
        nsamples,
        inference_fn=activation_equalization_iter,
        seqlen=seqlen,
        alpha=alpha)


@torch.no_grad()
def apply_weight_equalization(model, ref_kwargs, scale_computation_type='range'):
    graph_model = value_trace(model, value_args=ref_kwargs)
    EqualizeGraph(scale_computation_type=scale_computation_type).apply(graph_model)
