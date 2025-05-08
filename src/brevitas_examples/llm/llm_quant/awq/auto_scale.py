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

import inspect
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas_examples.llm.llm_quant.awq.graph import extract_sinks_scaling_factor
from brevitas_examples.llm.llm_quant.awq.utils.region import RegionAWQ

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def auto_scale_block(
        block: nn.Module,
        block_regions: List[RegionAWQ],
        block_kwargs: Dict[str, Any],
        input_feat: Dict[int, torch.Tensor]):

    if "use_cache" in block_kwargs:
        block_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(
            region_block: nn.Module,
            sinks: List[nn.Module],
            x: torch.Tensor,
            kwargs: Dict[str, Any] = {}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(region_block.parameters()).device)
        with quantization_status_manager(region_block,
                                         disable_act_quant=True,
                                         disable_weight_quant=True,
                                         disable_bias_quant=True):
            org_out = region_block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        scaling_factor = extract_sinks_scaling_factor(sinks)

        best_error = float('inf')
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = torch.reciprocal(scales / (scales.max() * scales.min()).sqrt())
            scaling_factor.data = scales
            # Capture quantized output from sinks
            quantization_status_manager_cm = quantization_status_manager(
                region_block,
                disable_act_quant=True,
                disable_weight_quant=True,
                disable_bias_quant=True)
            with quantization_status_manager_cm:
                # Restore quantization state of sinks
                for sink in sinks:
                    quantization_status_manager_cm.enable_module_quantization(module=sink)
                out = region_block(x * scales, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
        if best_ratio == -1:
            print(history)
            raise Exception

        # Set scaling factor back to identity
        scaling_factor.data = torch.ones_like(scaling_factor.data)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach().cpu()

    def _auto_get_scale(
            sinks: List[nn.Module],
            inp: torch.Tensor,
            region_block: Optional[nn.Module] = None,
            kwargs: Dict[str, Any] = {}):
        # block: if given, we will check the output diff of this module instead of layers
        if region_block is None:
            assert len(sinks) == 1
            region_block = sinks[0]
        return _search_module_scale(region_block, sinks, inp, kwargs)

    scales_dict = {}  # return the searched scales

    for region in block_regions:
        # Only scale non-orphan regions
        if len(region.srcs) > 0:
            # Decide which block_kwargs to propagate to the region block based on its forward signature
            region_block_args = [] if region.block is None else inspect.getfullargspec(
                region.block.forward).args
            kwargs = {arg: block_kwargs[arg] for arg in region_block_args if arg in block_kwargs}
            scales_dict[id(region)] = _auto_get_scale(
                sinks=[region.get_module_from_name(sink_name) for sink_name in region.sinks],
                inp=input_feat[id(region)],
                region_block=region.block,
                kwargs=kwargs,
            )
    return scales_dict


def apply_scale(
        block_regions: List[RegionAWQ],
        scales_dict: Dict[int, torch.Tensor],
        input_feat: Dict[int, torch.Tensor] = None) -> None:
    for region in block_regions:
        if len(region.srcs) > 0:
            region_id = id(region)
            scales = scales_dict[region_id]
            # Modify scaling factors appropiately
            scaling_factor = extract_sinks_scaling_factor([
                region.name_to_module[sink_name] for sink_name in region.sinks_names])
            scaling_factor.data = scales.to(scaling_factor.device)
            # Apply the scaling factor to the inputs
            if input_feat is not None:
                input_feat[region_id].mul_(scales.view(1, -1).to(input_feat[region_id].device))
