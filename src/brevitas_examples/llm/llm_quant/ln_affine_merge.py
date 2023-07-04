"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import torch
from torch import nn

from brevitas.fx import value_trace
from brevitas.graph.equalize import _is_reshaping_op
from brevitas.graph.equalize import _is_scale_invariant_module
from brevitas.graph.utils import get_module
from brevitas_examples.llm.llm_quant.run_utils import cast_to_float32


def replace_bias(next_module, new_bias):
    new_bias = new_bias.view(-1)
    if next_module.bias is not None:
        next_module.bias.data.copy_(new_bias)
    else:
        new_bias = new_bias.to(next_module.weight.device).to(next_module.weight.dtype)
        next_module.register_parameter('bias', torch.nn.Parameter(new_bias))


def _merge_ln(layer_norm, next_module, scale_bias_by_weight):
    if not layer_norm.elementwise_affine:
        return False
    if not isinstance(next_module, nn.Linear):
        return False
    view_shape = (1, -1)
    # Merge weight
    if scale_bias_by_weight:
        layer_norm.bias.data /= layer_norm.weight.data
    # We can't do an inplace update as some layers we merge into like lm_head might share the weight tensor
    scale = layer_norm.weight.data.view(view_shape).expand_as(next_module.weight)
    next_module.weight = torch.nn.Parameter(next_module.weight.clone() * scale)
    # Merge bias, new_bias includes the bias of next_module by going through its fwd
    inp = layer_norm.bias.data.view(view_shape)
    new_bias = next_module(inp)
    replace_bias(next_module, new_bias)
    return True


def merge_layernorm_affine_params(graph_model):
    merged_dict = {}
    merged_into_layers = []
    scaled_biases = set()
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if isinstance(module, nn.LayerNorm):
                for next in node.users:
                    while (_is_reshaping_op(next) or _is_scale_invariant_module(graph_model, next)):
                        next = node.next
                    if next.op == 'call_module':
                        next_module = get_module(graph_model, next.target)
                        scale_bias = node.target not in scaled_biases
                        merged = _merge_ln(module, next_module, scale_bias_by_weight=scale_bias)
                        if merged:
                            print(
                                f"{module.__class__.__name__} {node.target} merged into {next.target}."
                            )
                            merged_into_layers.append(next.target)
                            scaled_biases.add(node.target)
                        if module in merged_dict:
                            merged_dict[module] &= merged
                        else:
                            merged_dict[module] = merged
                    elif next.op == 'call_method' and next.target == 'size':
                        continue
                    else:
                        raise RuntimeError(
                            f"Unsupported user node {next.op} with target {next.target}. Disable LN affine merging."
                        )
    for module, merged in merged_dict.items():
        if merged:
            # We preserve weight and bias in case they are used to merge SmoothQuant scales in fx mode later on
            module.weight.data.fill_(1.)
            module.bias.data.fill_(0.)
        else:
            raise RuntimeError(
                f"Merged only into some users: {merged_dict}. Disable LN affine merging.")
    return merged_into_layers


@torch.no_grad()
def apply_layernorm_affine_merge(model, dtype, ref_kwargs):
    # We can't do fp16 tracing on CPU as many kernels are not implemented
    # So we have to cast to fp32 first, trace, apply merging, and then cast back
    with cast_to_float32(model, dtype):
        graph_model = value_trace(model, value_args=ref_kwargs)
        merge_layernorm_affine_params(graph_model)
