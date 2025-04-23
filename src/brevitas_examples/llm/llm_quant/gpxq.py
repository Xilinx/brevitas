# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from functools import partial

from accelerate.utils.operations import send_to_device
import torch
from tqdm import tqdm

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.magr import magr_mode
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.axe import A2GPFQ
from brevitas_examples.common.axe import A2GPTQ


def _gpxq_block_optimization_callback(block, gpxq, cached_args, cached_kwargs):
    for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
        for args, kwargs in zip(cached_args, cached_kwargs):
            args = send_to_device(args, 'cuda')
            kwargs = send_to_device(kwargs, 'cuda')
            block(*args, **kwargs)
        gpxq.update()


def _magr_block_optimization_callback(block, magr, cached_args, cached_kwargs):
    for args, kwargs in zip(cached_args, cached_kwargs):
        args = send_to_device(args, 'cuda')
        kwargs = send_to_device(kwargs, 'cuda')
        block(*args, **kwargs)
    magr.update()


@torch.no_grad()
def block_optimization(
        model,
        dataloader,
        block_name,
        context_manager_func,
        context_manager_kwargs,
        block_optimization_callback=_gpxq_block_optimization_callback):
    disable_quantization_cm = quantization_status_manager(
        model=model,
        disable_act_quant=not context_manager_kwargs.get('use_quant_activations', True),
        disable_weight_quant=False,
        disable_bias_quant=not context_manager_kwargs.get('use_quant_activations', True),
    )
    cache_state = model.config.use_cache
    model.config.use_cache = False
    blocks = recurse_getattr(model, block_name)
    first_block = blocks[0]
    cached_args, cached_kwargs = [], []

    # Intercept input to first block
    def intercept_input(module, args, kwargs):
        args = send_to_device(args, 'cpu')
        kwargs = send_to_device(kwargs, 'cpu')
        cached_args.append(args)
        cached_kwargs.append(kwargs)
        raise StopFwdException

    # Intercept output from block N-1 to set it as input to block N
    def intercept_output(module, args, kwargs, output):
        if isinstance(output, tuple):
            output = output[0]
        output = send_to_device(output, 'cpu')
        cached_args.append((output,))
        raise StopFwdException

    # Collect input to first block
    hook = first_block.register_forward_pre_hook(intercept_input, with_kwargs=True)
    with disable_quantization_cm:
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass
    hook.remove()

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            block_optimization_callback(block, gpxq, cached_args, cached_kwargs)

        if index < len(blocks) - 1:
            # Once the block is done, we need to update the input to the next block
            past_cached_args, past_cached_kwargs = deepcopy(cached_args), deepcopy(cached_kwargs)
            cached_args = []
            hook = block.register_forward_hook(intercept_output, with_kwargs=True)

            with disable_quantization_cm:
                for args, kwargs in zip(past_cached_args, past_cached_kwargs):
                    try:
                        args = send_to_device(args, 'cuda')
                        kwargs = send_to_device(kwargs, 'cuda')
                        block(*args, **kwargs)
                    except StopFwdException:
                        pass
            hook.remove()
    # Restore cache state
    model.config.use_cache = cache_state


@torch.no_grad()
def apply_gptq(
        model,
        dataloader,
        act_order=True,
        use_quant_activations=False,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        block_name=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gptq_class = partial(
            A2GPTQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gptq_class = GPTQ
    if block_name is not None:
        context_manager_kwargs = {
            'act_order': act_order,
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': create_weight_orig,
            'use_quant_activations': use_quant_activations,
            'gptq_class': gptq_class}
        block_optimization(model, dataloader, block_name, gptq_mode, context_manager_kwargs)
    else:
        with gptq_mode(model,
                       use_quant_activations=use_quant_activations,
                       group_of_parallel_layers=group_of_parallel_layers,
                       act_order=act_order,
                       create_weight_orig=create_weight_orig,
                       gptq_class=gptq_class) as gptq:
            gptq_model = gptq.model
            for _ in tqdm(range(gptq.num_layers)):
                for inps in dataloader:
                    gptq_model(**inps)
                gptq.update()


@torch.no_grad()
def apply_gpfq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gpfq_class = partial(
            A2GPFQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gpfq_class = GPFQ
    if block_name is not None:
        context_manager_kwargs = {
            'act_order': act_order,
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': True,
            'gpfq_class': gpfq_class}
        block_optimization(model, dataloader, block_name, gpfq_mode, context_manager_kwargs)
    else:
        with gpfq_mode(model,
                       act_order=act_order,
                       group_of_parallel_layers=group_of_parallel_layers,
                       create_weight_orig=True,
                       gpfq_class=gpfq_class) as gpfq:
            gpfq_model = gpfq.model
            for _ in tqdm(range(gpfq.num_layers)):
                for inps in dataloader:
                    gpfq_model(**inps)
                gpfq.update()


@torch.no_grad()
def apply_magr(
        model,
        dataloader,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=0.01,
        num_steps=200):
    if block_name is not None:
        context_manager_kwargs = {
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': create_weight_orig,
            'alpha': alpha,
            'num_steps': num_steps}
        block_optimization(
            model,
            dataloader,
            block_name,
            magr_mode,
            context_manager_kwargs,
            block_optimization_callback=_magr_block_optimization_callback)
    else:
        with magr_mode(model,
                       group_of_parallel_layers=group_of_parallel_layers,
                       create_weight_orig=create_weight_orig,
                       num_steps=num_steps,
                       alpha=alpha) as magr:
            magr_model = magr.model
            for inps in tqdm(dataloader, desc="Calculating covariances..."):
                magr_model(**inps)
            magr.update()
