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

from argparse import Namespace
from collections import defaultdict
import functools
import gc
from typing import List, Optional

import torch
import torch.nn as nn
import tqdm
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM

from brevitas.graph.calibrate import disable_enable_quantization
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import Region
from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.llm.llm_quant.awq.auto_clip import apply_clip
from brevitas_examples.llm.llm_quant.awq.auto_clip import auto_clip_block
from brevitas_examples.llm.llm_quant.awq.auto_scale import apply_scale
from brevitas_examples.llm.llm_quant.awq.auto_scale import auto_scale_block
from brevitas_examples.llm.llm_quant.awq.graph import EqualizeAWQ
from brevitas_examples.llm.llm_quant.awq.graph import initialize_awq_region
from brevitas_examples.llm.llm_quant.awq.utils.calib_data import get_calib_dataset
from brevitas_examples.llm.llm_quant.awq.utils.region import RegionAWQ
from brevitas_examples.llm.llm_quant.awq.utils.region import retrieve_block_awq_regions

__all__ = ["run_awq"]


# TODO: Reuse function
def get_blocks(model: nn.Module) -> nn.ModuleList:
    if isinstance(model, LlamaForCausalLM):
        blocks = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        blocks = model.model.decoder.layers
    else:
        raise NotImplementedError(type(model))
    return blocks


def _regions_to_block(blocks: List[nn.Module], regions: List[RegionAWQ]) -> List[List[RegionAWQ]]:
    regions_per_block = [[] for _ in range(len(blocks))]
    for region in regions:
        for i, block in enumerate(blocks):
            if all(any(m_region is m_block
                       for m_block in block.modules())
                   for m_region in region.name_to_module.values()):
                regions_per_block[i].append(region)
                break
    # Verify that the regions were assigned correctly
    assert all(map(lambda x : len(x) == len(regions_per_block[0]), regions_per_block)), "The number of regions assigned to each block is not constant."
    return regions_per_block


def prepare_awq_regions(model: nn.Module, blocks: List[nn.Module],
                        regions: List[Region]) -> List[List[RegionAWQ]]:
    # Fix regions to point to quantized modules
    for region in regions:
        for module_name in region.name_to_module.keys():
            region.name_to_module[module_name] = recurse_getattr(model, module_name)
    regions = [initialize_awq_region(model, region) for region in regions]
    regions_per_block = _regions_to_block(blocks, regions)
    return regions_per_block


@torch.no_grad()
def run_awq(
    model: nn.Module,
    tokenizer,
    args: Namespace,
    regions: Optional[List[Region]] = None,
    n_samples: int = 512,
    seqlen: int = 512,
    auto_scale: bool = True,
    mse_range: bool = True,
    calib_data: str = "pileval",
):
    model.cuda()
    # TODO: Reuse computation
    blocks = get_blocks(model)
    # Prepare AWQ regions
    if regions is not None:
        regions_per_block = prepare_awq_regions(model, blocks, regions)
    else:
        regions_per_block = [retrieve_block_awq_regions(block) for block in blocks]
        # Apply the regions
        eq = EqualizeAWQ(
            weight_group_size=args.weight_group_size
            if args.weight_quant_granularity == 'per_group' else None,
            add_parametrizations_inplace=True,
        )
        # TODO: Consider using a more readable alternative to sum(regions_per_block, [])
        model, _, _ = eq.apply(model=model, regions=sum(regions_per_block, []))

    samples = get_calib_dataset(
        data=calib_data, tokenizer=tokenizer, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    block_kwargs = {}

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            block_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    blocks[0] = Catcher(blocks[0])
    try:
        with disable_enable_quantization(model, disable_quant=True):
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    blocks[0] = blocks[0].module  # restore
    inps = inps[0]

    gc.collect()
    torch.cuda.empty_cache()
    # solve layer by layer
    for i in tqdm.tqdm(range(len(blocks)), desc="Running AWQ..."):
        block = blocks[i]
        block_regions = regions_per_block[i]

        # firstly, get input features of all linear blocks
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for region in block_regions:
            sink = region.repr_sink
            handles.append(
                sink.register_forward_hook(
                    functools.partial(cache_input_hook, name=id(region), feat_dict=input_feat)))
        inps = inps.to(next(block.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        with disable_enable_quantization(model, disable_quant=True):
            inps = block(inps, **block_kwargs)[0]
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_dict = auto_scale_block(
                block,
                block_regions,
                block_kwargs,
                input_feat=input_feat,
            )
            apply_scale(block_regions=block_regions, scales_dict=scales_dict, input_feat=input_feat)
        # Fuse the scaling and clipping parametrizations
        block = fuse_parametrizations(block)
        if mse_range:
            clip_dict = auto_clip_block(
                block_regions=block_regions,
                input_feat=input_feat,
            )

            apply_clip(block_regions=block_regions, clip_dict=clip_dict)

        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    model.cpu()
