"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import warnings

import torch

from brevitas.fx.brevitas_tracer import value_trace
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizeGraph
from brevitas_examples.llm.llm_quant.run_utils import apply_layer_ptq_fn
from brevitas_examples.llm.llm_quant.run_utils import cast_to_float32


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
def apply_act_equalization(
        model,
        dtype,
        act_equalization_type,
        dataloader,
        nsamples,
        seqlen=2048,
        alpha=0.5,
        ref_kwargs=None):
    if act_equalization_type == 'layerwise':
        apply_layer_ptq_fn(
            model,
            dataloader,
            nsamples,
            inference_fn=activation_equalization_iter,
            seqlen=seqlen,
            alpha=alpha)
    elif act_equalization_type == 'fx':
        assert ref_kwargs is not None, "Ref kwargs required to perform tracing and lift the model into FX."
        # We can't do fp16 tracing on CPU as many kernels are not implemented
        # So we have to cast to fp32 first, trace, apply equalization, and then cast back
        with cast_to_float32(model, dtype):
            graph_model = value_trace(model, value_args=ref_kwargs)
            # TODO this is currently running on CPU. We need Accelerate or a TorchDispatchMode
            # or an FX interpreter to run it on GPU
            warnings.warn(
                "FX mode activation equalization currently runs on CPU, expect it to be slow for large models."
            )
            with activation_equalization_mode(graph_model,
                                              alpha,
                                              add_mul_node=False,
                                              layerwise=False):
                for input_ids in dataloader:
                    graph_model(input_ids=input_ids)
    else:
        raise RuntimeError(f"{act_equalization_type} not supported.")


@torch.no_grad()
def apply_weight_equalization(model, dtype, ref_kwargs, scale_computation_type='range'):
    # We can't do fp16 tracing on CPU as many kernels are not implemented
    # So we have to cast to fp32 first, trace, apply equalization, and then cast back
    with cast_to_float32(model, dtype):
        graph_model = value_trace(model, value_args=ref_kwargs)
        EqualizeGraph(scale_computation_type=scale_computation_type).apply(graph_model)
