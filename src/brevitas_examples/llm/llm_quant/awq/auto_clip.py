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

import torch
import torch.nn as nn

from .quantizer import pseudo_quantize_tensor

__all__ = ["auto_clip_block"]


# weight quantization
@torch.no_grad()
def auto_clip_layer(w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

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
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):

    named_linears = {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_"]]):
            continue
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config)
        clip_list.append((name, max_val))
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    from .utils.module import get_op_by_name
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
