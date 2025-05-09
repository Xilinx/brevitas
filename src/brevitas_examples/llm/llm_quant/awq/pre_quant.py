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
from typing import Dict, List, Optional, Tuple

from accelerate.utils.operations import send_to_device
import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.equalize import EqualizationIndexes
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.awq.auto_clip import apply_clip
from brevitas_examples.llm.llm_quant.awq.auto_clip import auto_clip_block
from brevitas_examples.llm.llm_quant.awq.auto_scale import apply_scale
from brevitas_examples.llm.llm_quant.awq.auto_scale import auto_scale_block
from brevitas_examples.llm.llm_quant.awq.graph import EqualizeAWQ
from brevitas_examples.llm.llm_quant.awq.utils.region import RegionAWQ
from brevitas_examples.llm.llm_quant.awq.utils.region import retrieve_block_awq_regions
from brevitas_examples.llm.llm_quant.data_utils import DatasetToDevice

__all__ = ["apply_awq"]


def get_blocks_attribute(model: nn.Module) -> str:
    if model.__class__.__name__ == "LlamaForCausalLM":
        return "model.layers"
    elif model.__class__.__name__ == "OPTForCausalLM":
        return "model.decoder.layers"

    raise NotImplementedError(f"Blocks attribute for model {type(model)} is unknown")


def _retrieve_per_block_regions(blocks: List[nn.Module]) -> List[List[RegionAWQ]]:
    regions_per_block = [retrieve_block_awq_regions(block) for block in blocks]
    # Incorporate the orphan regions
    _add_orphan_regions(blocks, regions_per_block)
    return regions_per_block


def _add_orphan_regions(blocks: List[nn.Module], regions_per_block: List[List[RegionAWQ]]) -> None:
    # Find linears which are not registered as sinks of any region
    for i, block in enumerate(blocks):
        for name, module in block.named_modules():
            if (isinstance(module, nn.Linear) and
                    not any(any(module is region.get_module_from_name(sink_name)
                                for sink_name in region.sinks)
                            for region in regions_per_block[i])):
                # Create region for an orphan sink
                eq_indexes = EqualizationIndexes(0, module.weight.shape[0], 0)
                regions_per_block[i].append(
                    RegionAWQ(sinks={name: eq_indexes}, name_to_module={name: module}))


# Hook to capture inputs to module
def intercept_input(
        module: nn.Module,
        args: Tuple,
        kwargs: Dict,
        args_list: List[Tuple],
        kwargs_list: Optional[List[Tuple]],
        raise_exception: bool = True):
    if isinstance(args, tuple):
        args = args[0]
    args = send_to_device(args, 'cpu')
    args_list.append(args)
    if kwargs_list is not None:
        kwargs = send_to_device(kwargs, 'cpu')
        kwargs_list.append(kwargs)
    if raise_exception:
        raise StopFwdException


@torch.no_grad()
def apply_awq(
    model: nn.Module,
    tokenizer,
    calibration_loader: DatasetToDevice,
    args: Namespace,
    auto_scale: bool = True,
    mse_range: bool = True,
):
    # Cache needs to be disabled for training
    cache_state = model.config.use_cache
    model.config.use_cache = False
    # Retrive model blocks
    blocks = recurse_getattr(
        model,
        get_blocks_attribute(model) if args.gpxq_block_name is None else args.gpxq_block_name)

    # Concatenate input_ids across the batch dimension
    samples = torch.cat(list(map(lambda sample: sample["input_ids"], calibration_loader)), dim=0)

    first_block = blocks[0]
    cached_args, cached_kwargs = [], []

    # Capture inputs to the first block
    hook = first_block.register_forward_pre_hook(
        functools.partial(
            intercept_input,
            args_list=cached_args,
            kwargs_list=cached_kwargs,
            raise_exception=True,
        ),
        with_kwargs=True)
    model = offload_model(model)
    with quantization_status_manager(model,
                                     disable_act_quant=True,
                                     disable_weight_quant=True,
                                     disable_bias_quant=True):
        try:
            model(samples)
        except StopFwdException:
            pass
    hook.remove()
    remove_hooks(model)

    # Retrieve AWQ regions
    regions_per_block = _retrieve_per_block_regions(blocks)
    # Add scaling modules for optimization
    if auto_scale:
        eq = EqualizeAWQ()
        model, _, _ = eq.apply(model=model, regions=sum(regions_per_block, []))

    # Prepare inputs
    inps = cached_args[0]
    block_kwargs = cached_kwargs[0]
    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        block.cuda()
        device = next(block.parameters()).device
        block_regions = regions_per_block[index]

        input_feat = defaultdict(list)
        hooks = []
        for region in block_regions:
            sink = region.repr_sink
            hooks.append(
                sink.register_forward_pre_hook(
                    functools.partial(
                        intercept_input,
                        args_list=input_feat[id(region)],
                        kwargs_list=None,
                        raise_exception=False,
                    ),
                    with_kwargs=True))

        inps = inps.to(device)  # in case multi-gpu
        block_kwargs = send_to_device(block_kwargs, device)
        # get output as next layer's input
        with quantization_status_manager(model,
                                         disable_act_quant=True,
                                         disable_weight_quant=True,
                                         disable_bias_quant=True):
            inps = block(inps, **block_kwargs)[0]
        for hook in hooks:
            hook.remove()

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
            # Fuse the scaling parametrizations
            block = fuse_parametrizations(block)
        if mse_range:
            clip_dict = auto_clip_block(
                block_regions=block_regions,
                input_feat=input_feat,
            )
            apply_clip(block_regions=block_regions, clip_dict=clip_dict)

        del input_feat
        block.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    # Restore cache state
    model.config.use_cache = cache_state
