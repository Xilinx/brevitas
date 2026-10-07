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
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.magr import magr_mode
from brevitas.graph.qronos import Qronos
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.axe import a2gpfq_mode
from brevitas_examples.common.axe import a2gptq_mode


def _gptq_block_optimization_callback(
        block,
        gptq,
        cached_args,
        cached_kwargs,
        _unused_cached_args,
        _unused_cached_kwargs,
        _unused_context_manager):
    for _ in tqdm(range(gptq.num_layers), desc="Layers", leave=False):
        for args, kwargs in zip(cached_args, cached_kwargs):
            args = send_to_device(args, 'cuda')
            kwargs = send_to_device(kwargs, 'cuda')
            block(*args, **kwargs)
        gptq.update()


def _magr_block_optimization_callback(
        block,
        magr,
        cached_args,
        cached_kwargs,
        _unused_cached_args,
        _unused_cached_kwargs,
        _unused_context_manager):
    for args, kwargs in zip(cached_args, cached_kwargs):
        args = send_to_device(args, 'cuda')
        kwargs = send_to_device(kwargs, 'cuda')
        block(*args, **kwargs)
    magr.update()


def _gpfq_or_qronos_block_optimization_callback(
        block,
        gpxq,
        float_cached_args,
        float_cached_kwargs,
        quant_cached_args,
        quant_cached_kwargs,
        disable_quantization_cm):
    for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
        for quant_args, quant_kwargs, float_args, float_kwargs in zip(
                quant_cached_args,
                quant_cached_kwargs,
                float_cached_args,
                float_cached_kwargs):
            # Run the quantized pass first. GPFQ and Qronos store its input.
            quant_args = send_to_device(quant_args, 'cuda')
            quant_kwargs = send_to_device(quant_kwargs, 'cuda')
            gpxq.model(*quant_args, **quant_kwargs)
            # Run the float pass second. GPFQ and Qronos use the input pair to update G.
            with disable_quantization_cm:
                float_args = send_to_device(float_args, 'cuda')
                float_kwargs = send_to_device(float_kwargs, 'cuda')
                gpxq.model(*float_args, **float_kwargs)
        # Update after all input pairs are available for the current layer.
        gpxq.update()


def _intercept_input(module, args, kwargs, cached_args, cached_kwargs):
    """Cache a block input on the CPU and stop the forward pass."""
    cached_args.append(send_to_device(args, 'cpu'))
    cached_kwargs.append(send_to_device(kwargs, 'cpu'))
    raise StopFwdException


def _intercept_output(module, args, kwargs, output, cached_args):
    """Cache a block output on the CPU and stop the forward pass."""
    if isinstance(output, tuple):
        output = output[0]
    cached_args.append((send_to_device(output, 'cpu'),))
    raise StopFwdException


def _block_optimization(
        model,
        dataloader,
        block_name,
        context_manager_func,
        context_manager_kwargs,
        block_optimization_callback=_gptq_block_optimization_callback,
        reset_float_cache_every=None):

    if reset_float_cache_every is not None:
        if not isinstance(reset_float_cache_every, int) or reset_float_cache_every <= 0:
            raise ValueError("reset_float_cache_every must be None or a positive integer.")

    # Algorithms such as GPFQ and Qronos aim to solve the mismatched objective by using
    # float inputs and (possibly quantized) inputs from the previously quantized layers.
    use_quant_activations = context_manager_kwargs.get('use_quant_activations', True)
    solves_mismatched_objective = context_manager_func.solves_mismatched_objective
    if solves_mismatched_objective and not use_quant_activations:
        raise ValueError("Mismatched objective requires use_quant_activations=True.")

    disable_quantization_cm = quantization_status_manager(
        model=model,
        disable_act_quant=solves_mismatched_objective or not use_quant_activations,
        disable_weight_quant=solves_mismatched_objective,
        disable_bias_quant=solves_mismatched_objective or not use_quant_activations,
        is_training=False)

    cache_state = model.config.use_cache
    model.config.use_cache = False
    blocks = recurse_getattr(model, block_name)
    first_block = blocks[0]

    # Cached inputs from the model under the context manager.
    float_cached_args, float_cached_kwargs = [], []
    # Cached inputs from the quantized model if solves_mismatched_objective is True.
    quant_cached_args, quant_cached_kwargs = [], []

    if solves_mismatched_objective:
        # Cache inputs from the quantized model first.
        hook = first_block.register_forward_pre_hook(
            partial(
                _intercept_input, cached_args=quant_cached_args, cached_kwargs=quant_cached_kwargs),
            with_kwargs=True)
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass
        hook.remove()

    hook = first_block.register_forward_pre_hook(
        partial(_intercept_input, cached_args=float_cached_args, cached_kwargs=float_cached_kwargs),
        with_kwargs=True)
    with disable_quantization_cm:
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass
    hook.remove()

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        # Create a new context manager for the current block.
        # Disabling quantization for the current block is cheaper than disabling it for the entire model.
        disable_quantization_cm = quantization_status_manager(
            model=block,
            disable_act_quant=solves_mismatched_objective or not use_quant_activations,
            disable_weight_quant=solves_mismatched_objective,
            disable_bias_quant=solves_mismatched_objective or not use_quant_activations,
            is_training=False)
        # The context manager installs hooks for the current block.
        # The callback runs the passes expected by these hooks.
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            block_optimization_callback(
                block,
                gpxq,
                float_cached_args,
                float_cached_kwargs,
                quant_cached_args,
                quant_cached_kwargs,
                disable_quantization_cm)

        if index < len(blocks) - 1:
            # Once the block is done, we need to update the input to the next block
            if solves_mismatched_objective:
                past_quant_cached_args = deepcopy(quant_cached_args)
                quant_cached_args = []
                hook = block.register_forward_hook(
                    partial(_intercept_output, cached_args=quant_cached_args), with_kwargs=True)
                for args, kwargs in zip(past_quant_cached_args, quant_cached_kwargs):
                    try:
                        args = send_to_device(args, 'cuda')
                        kwargs = send_to_device(kwargs, 'cuda')
                        block(*args, **kwargs)
                    except StopFwdException:
                        pass
                hook.remove()

                if reset_float_cache_every is not None and (index +
                                                            1) % reset_float_cache_every == 0:
                    float_cached_args = deepcopy(quant_cached_args)
                    float_cached_kwargs = deepcopy(quant_cached_kwargs)
                    continue

            past_float_cached_args = deepcopy(float_cached_args)
            past_float_cached_kwargs = deepcopy(float_cached_kwargs)
            float_cached_args = []
            hook = block.register_forward_hook(
                partial(_intercept_output, cached_args=float_cached_args), with_kwargs=True)
            with disable_quantization_cm:
                for args, kwargs in zip(past_float_cached_args, past_float_cached_kwargs):
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
        max_accumulator_tile_size=None,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    context_manager_kwargs = {
        'act_order': act_order,
        'group_of_parallel_layers': group_of_parallel_layers,
        'create_weight_orig': create_weight_orig,
        'use_quant_activations': use_quant_activations,
        'device': buffer_device,
        'dtype': buffer_dtype}
    context_manager_func = gptq_mode
    if max_accumulator_bit_width is not None:
        context_manager_func = a2gptq_mode
        context_manager_kwargs.update(
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    if block_name is not None:
        _block_optimization(
            model, dataloader, block_name, context_manager_func, context_manager_kwargs)
    else:
        with context_manager_func(model, **context_manager_kwargs) as gptq:
            gptq_model = gptq.model
            for _ in tqdm(range(gptq.num_layers)):
                for inps in dataloader:
                    gptq_model(**inps)
                gptq.update()


def _apply_gpfq_or_qronos(
        model,
        dataloader,
        act_order=True,
        block_name=None,
        group_of_parallel_layers=None,
        algorithm_impl=GPFQ,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None,
        device='cpu',
        dtype=torch.float32):
    """
    This wraps gpfq_mode, which can be used for any layerwise PTQ algorithm that
    optimizes the mismatched objective function || XW - \tilde{X}Q ||, where
    Q is the quantized weights and \tilde{X} are the (potentially quantized)
    activations resulting from the previously quantized layers.

    See https://arxiv.org/abs/2505.11695 for more!
    """
    context_manager_kwargs = {
        'act_order': act_order,
        'group_of_parallel_layers': group_of_parallel_layers,
        'create_weight_orig': True,
        'algorithm_impl': algorithm_impl,
        'device': device,
        'dtype': dtype}
    context_manager_func = gpfq_mode
    if max_accumulator_bit_width is not None:
        context_manager_func = a2gpfq_mode
        context_manager_kwargs.update(
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    if block_name is not None:
        _block_optimization(
            model,
            dataloader,
            block_name,
            context_manager_func,
            context_manager_kwargs,
            block_optimization_callback=_gpfq_or_qronos_block_optimization_callback,
            reset_float_cache_every=1)
    else:
        disable_quantization_cm = quantization_status_manager(
            model=model,
            disable_act_quant=True,
            disable_weight_quant=True,
            disable_bias_quant=True,
            is_training=False)
        # The context manager installs hooks used by GPFQ or Qronos.
        # Run both passes before update consumes their collected inputs.
        with context_manager_func(model, **context_manager_kwargs) as algo:
            for _ in tqdm(range(algo.num_layers)):
                for inps in dataloader:
                    # Run the quantized pass first. GPFQ and Qronos store its input.
                    algo.model(**inps)
                    # Run the float pass second. GPFQ and Qronos use the input pair to update G.
                    with disable_quantization_cm:
                        algo.model(**inps)
                algo.update()


@torch.no_grad()
def apply_gpfq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    # Use paired quantized and float passes to correct errors from previous layers.
    _apply_gpfq_or_qronos(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=GPFQ,
        max_accumulator_bit_width=max_accumulator_bit_width,
        max_accumulator_tile_size=max_accumulator_tile_size,
        device=buffer_device,
        dtype=buffer_dtype)


@torch.no_grad()
def apply_qronos(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=1e-6,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    assert alpha > 0, "Error: alpha needs to be strictly positive"
    # Use paired quantized and float passes to correct errors from previous layers.
    _apply_gpfq_or_qronos(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=partial(Qronos, alpha=alpha),
        device=buffer_device,
        dtype=buffer_dtype)


@torch.no_grad()
def apply_magr(
        model,
        dataloader,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=0.01,
        num_steps=200,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    if block_name is not None:
        context_manager_kwargs = {
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': create_weight_orig,
            'alpha': alpha,
            'num_steps': num_steps,
            'device': buffer_device,
            'dtype': buffer_dtype}
        _block_optimization(
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
                       alpha=alpha,
                       device=buffer_device,
                       dtype=buffer_dtype) as magr:
            magr_model = magr.model
            for inps in tqdm(dataloader, desc="Calculating covariances..."):
                magr_model(**inps)
            magr.update()
