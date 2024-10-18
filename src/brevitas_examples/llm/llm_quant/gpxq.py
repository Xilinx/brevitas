# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from functools import partial

from accelerate.utils.operations import send_to_device
import torch
from tqdm import tqdm

from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gpfq import GPFQv2
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import StopFwdException
from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.axe import A2GPFQ
from brevitas_examples.common.axe import A2GPTQ


@torch.no_grad()
def block_optimization(model, dataloader, block_name, context_manager_func, context_manager_kwargs):
    disable_quant_inference = DisableEnableQuantization()
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
    if not context_manager_kwargs.get('use_quant_activations', True):
        return_quant_tensor_state = disable_return_quant_tensor(model)
        disable_quant_inference.disable_act_quantization(model, is_training=model.training)
        disable_quant_inference.disable_bias_quantization(model, is_training=model.training)

    hook = first_block.register_forward_pre_hook(intercept_input, with_kwargs=True)
    for inps in dataloader:
        try:
            model(**inps)
        except StopFwdException:
            pass
    hook.remove()

    if not context_manager_kwargs.get('use_quant_activations', True):
        disable_quant_inference.enable_act_quantization(model, is_training=model.training)
        disable_quant_inference.enable_bias_quantization(model, is_training=model.training)
        restore_return_quant_tensor(model, return_quant_tensor_state)

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
                for args, kwargs in zip(cached_args, cached_kwargs):
                    args = send_to_device(args, 'cuda')
                    kwargs = send_to_device(kwargs, 'cuda')
                    block(*args, **kwargs)
                gpxq.update()

        if index < len(blocks) - 1:
            # Once the block is done, we need to update the input to the next block
            past_cached_args, past_cached_kwargs = deepcopy(cached_args), deepcopy(cached_kwargs)
            cached_args = []
            hook = block.register_forward_hook(intercept_output, with_kwargs=True)

            if not context_manager_kwargs.get('use_quant_activations', True):
                return_quant_tensor_state = disable_return_quant_tensor(model)
                disable_quant_inference.disable_act_quantization(model, is_training=model.training)
                disable_quant_inference.disable_bias_quantization(model, is_training=model.training)

            for args, kwargs in zip(past_cached_args, past_cached_kwargs):
                try:
                    args = send_to_device(args, 'cuda')
                    kwargs = send_to_device(kwargs, 'cuda')
                    block(*args, **kwargs)
                except StopFwdException:
                    pass

            if not context_manager_kwargs.get('use_quant_activations', True):
                disable_quant_inference.enable_act_quantization(model, is_training=model.training)
                disable_quant_inference.enable_bias_quantization(model, is_training=model.training)
                restore_return_quant_tensor(model, return_quant_tensor_state)

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
        max_accumulator_tile_size=128):
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
        max_accumulator_tile_size=128):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gpfq_class = partial(
            A2GPFQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gpfq_class = GPFQv2
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
