"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Adapted from https://github.com/mit-han-lab/llm-awq, released under the following LICENSE:

MIT License

Copyright (c) 2023 MIT HAN Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gc
from typing import Dict, List

import torch
import torch.nn as nn

from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas_examples.llm.llm_quant.awq.utils.region import RegionAWQ

__all__ = ["auto_clip_block", "apply_clip"]


def _get_greatest_po2_divisor(num: int):
    return (num & (~(num - 1)))


# Auxiliar method to retrieve properties of the weight quantizer
def _get_weight_quant_properties(sink: nn.Module, oc_batch_size: int = 256):
    if isinstance(
            sink.weight_quant,
        (GroupwiseWeightQuantProxyFromInjector, GroupwiseWeightFloatQuantProxyFromInjector)):
        num_output_channels, num_groups, group_size = sink.weight_quant.tensor_quant.int_quant.input_view_impl.expanded_groupwise_shape
        oc_batch_size = oc_batch_size if num_output_channels % oc_batch_size == 0 else _get_greatest_po2_divisor(
            num_output_channels)
        quant_injector_properties = {
            "stats_output_shape": (num_output_channels, num_groups, 1),
            "expanded_groupwise_shape": (num_output_channels, num_groups, group_size),}
        batch_quant_injector_properties = {
            "stats_output_shape": (oc_batch_size, num_groups, 1),
            "expanded_groupwise_shape": (oc_batch_size, num_groups, group_size),}
    elif isinstance(sink.weight_quant,
                    (WeightQuantProxyFromInjector, WeightFloatQuantProxyFromInjector)):
        num_output_channels, num_groups, group_size = sink.weight.shape[0], 1, sink.weight.shape[1]
        oc_batch_size = oc_batch_size if num_output_channels % oc_batch_size == 0 else _get_greatest_po2_divisor(
            num_output_channels)
        quant_injector_properties = {
            "scaling_per_output_channel_shape": (num_output_channels, 1),}
        batch_quant_injector_properties = {
            "scaling_per_output_channel_shape": (oc_batch_size, 1),}
    else:
        raise ValueError(f"{type(sink.weight_quant)} not supported")
    return group_size, oc_batch_size, quant_injector_properties, batch_quant_injector_properties


# weight quantization
@torch.no_grad()
def auto_clip_layer(
        sink: nn.Module, input_feat: torch.Tensor, n_grid=20, max_shrink=0.5, n_sample_token=512):
    # Retrieve group size and injector properties to clip in weight batches
    group_size, oc_batch_size, quant_injector_properties, batch_quant_injector_properties = _get_weight_quant_properties(sink)

    w = sink.weight.data
    assert w.dim() == 2
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    # Enable quantizing a weight batch to prevent OOM
    sink.weight_quant.quant_injector = sink.weight_quant.quant_injector.let(
        **batch_quant_injector_properties)
    sink.weight_quant.init_tensor_quant(preserve_state_dict=True)

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = sink.weight_quant(cur_w.view(
                oc_batch_size, -1)).value.reshape(*[oc_batch_size, 1, -1, group_size])
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    # Restore initial configuration for the quant_injector
    sink.weight_quant.quant_injector = sink.weight_quant.quant_injector.let(
        **quant_injector_properties)
    sink.weight_quant.init_tensor_quant(preserve_state_dict=True)
    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1).detach().cpu()


@torch.no_grad()
def auto_clip_block(block_regions: List[RegionAWQ], input_feat: Dict[int, torch.Tensor]) -> None:
    clip_dict = {}
    for region in block_regions:
        for name in region.sinks_names:
            sink = region.name_to_module[name]
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue
            max_val = auto_clip_layer(sink, input_feat[id(region)])
            clip_dict[name] = max_val
    return clip_dict


@torch.no_grad()
def apply_clip(block_regions: List[RegionAWQ], clip_dict: Dict[int, torch.Tensor]) -> None:
    # Set the clipping values to the optimal values found
    for region in block_regions:
        for name in region.sinks_names:
            if name in clip_dict:
                sink = region.name_to_module[name]
                sink.cuda()
                max_val = clip_dict[name].to(sink.weight.device)
                org_shape = sink.weight.shape
                sink.weight.data = sink.weight.data.reshape(*max_val.shape[:2], -1)
                sink.weight.data = torch.clamp(sink.weight.data, -max_val, max_val)
                sink.weight.data = sink.weight.data.reshape(org_shape)
                sink.cpu()
