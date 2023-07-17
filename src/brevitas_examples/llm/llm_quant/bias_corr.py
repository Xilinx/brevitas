"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch

from brevitas.graph.calibrate import bias_correction_mode
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn


@torch.no_grad()
def bias_corr_iter(curr_layer, inps, outs, cached_values):
    curr_layer = curr_layer.cuda()
    with bias_correction_mode(curr_layer):
        for j in range(len(inps)):
            inp = inps[j].unsqueeze(0).cuda()
            curr_out = curr_layer(inp, **cached_values)[0]
            outs[j] = curr_out
    curr_layer.cpu()
    return outs


@torch.no_grad()
def apply_bias_correction(model, dataloader, nsamples, seqlen=2048):
    apply_layer_ptq_fn(model, dataloader, nsamples, inference_fn=bias_corr_iter, seqlen=seqlen)
