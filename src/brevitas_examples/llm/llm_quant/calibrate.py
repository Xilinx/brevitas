"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn
from brevitas_examples.optimum.utils import offload_model
from brevitas_examples.optimum.utils import remove_hooks


@torch.no_grad()
def calibration_iter(curr_layer, inps, outs, cached_values):
    curr_layer = curr_layer.cuda()
    with calibration_mode(curr_layer):
        for j in range(len(inps)):
            inp = inps[j].unsqueeze(0).cuda()
            curr_out = curr_layer(inp, **cached_values)[0]
            outs[j] = curr_out
    curr_layer.cpu()
    return outs


@torch.no_grad()
def apply_calibration(model, dataloader):
    model = offload_model(model)
    with calibration_mode(model):
        for inps in tqdm(dataloader):
            inps = {k: v.cuda() for (k, v) in inps.items()}
            model(**inps)
    # Remove all accelerate hooks
    remove_hooks(model)
