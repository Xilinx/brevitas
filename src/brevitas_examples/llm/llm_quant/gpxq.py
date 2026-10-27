# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Optional
from typing import Type

from accelerate.utils.operations import send_to_device
import torch
from tqdm import tqdm

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.magr import magr_mode
from brevitas.graph.piso import PiSOInterleavedGPTQ
from brevitas.graph.piso import PiSOInterleavedQronos
from brevitas.graph.piso import PiSOMixin
from brevitas.graph.qronos import Qronos
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.axe import a2gpfq_mode
from brevitas_examples.common.axe import a2gptq_mode


def _layer_optimization_callback(gpxq, passes):
    """Calibrate every layer of a block on the cached inputs, then update the layer.

    passes holds one (cached_args, cached_kwargs, pass_manager) per forward pass of
    each batch, in the order they run. GPFQ and Qronos run the quantized pass first,
    which stores its input, and the float pass second, which pairs it with that input.
    """
    pass_managers = [pass_manager for _, _, pass_manager in passes]
    for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
        for batch in zip(*(zip(args, kwargs) for args, kwargs, _ in passes)):
            for (args, kwargs), pass_manager in zip(batch, pass_managers):
                args = send_to_device(args, 'cuda')
                kwargs = send_to_device(kwargs, 'cuda')
                with pass_manager:
                    gpxq.model(*args, **kwargs)
        gpxq.update()


def _magr_block_optimization_callback(magr, passes):
    cached_args, cached_kwargs, _ = passes[0]
    for args, kwargs in zip(cached_args, cached_kwargs):
        args = send_to_device(args, 'cuda')
        kwargs = send_to_device(kwargs, 'cuda')
        magr.model(*args, **kwargs)
    magr.update()


def _gpxq_pass_managers(module, solves_mismatched_objective, use_quant_activations):
    """Quantization state of the forward passes that feed one block of a GPxQ algorithm.

    An algorithm that solves the mismatched objective (GPFQ, Qronos) runs a quantized
    pass and then a float pass with all quantization disabled. Any other algorithm runs
    a single pass, which disables activation and bias quantization unless
    use_quant_activations is set.
    """
    if solves_mismatched_objective:
        return [
            nullcontext(),
            quantization_status_manager(
                model=module,
                disable_act_quant=True,
                disable_weight_quant=True,
                disable_bias_quant=True)]
    return [
        quantization_status_manager(
            model=module,
            disable_act_quant=not use_quant_activations,
            disable_bias_quant=not use_quant_activations)]


def _default_pass_managers(context_manager_func, context_manager_kwargs):
    solves_mismatched_objective = context_manager_func.solves_mismatched_objective
    use_quant_activations = context_manager_kwargs.get('use_quant_activations', True)
    if solves_mismatched_objective and not use_quant_activations:
        raise ValueError("Mismatched objective requires use_quant_activations=True.")
    return partial(
        _gpxq_pass_managers,
        solves_mismatched_objective=solves_mismatched_objective,
        use_quant_activations=use_quant_activations)


def _calibrate_model(
        model, dataloader, context_manager_func, context_manager_kwargs, pass_managers=None):
    """Calibrate every layer of the whole model on the dataloader, then update each layer."""
    if pass_managers is None:
        pass_managers = _default_pass_managers(context_manager_func, context_manager_kwargs)
    passes = pass_managers(model)
    with context_manager_func(model, **context_manager_kwargs) as algo:
        for _ in tqdm(range(algo.num_layers)):
            for inps in dataloader:
                for pass_manager in passes:
                    with pass_manager:
                        algo.model(**inps)
            algo.update()


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


# TODO: consider refactoring this into a class where GPTQ, MagR, GPFQ, Qronos, etc.
# can each inherit from and implement/override their own logic.
def _block_optimization(
        model,
        dataloader,
        block_name,
        context_manager_func,
        context_manager_kwargs,
        block_optimization_callback=_layer_optimization_callback,
        reset_float_cache_every=None,
        pass_managers=None):
    """Run a layer-wise algorithm block by block, propagating the block inputs.

    pass_managers(module) returns the quantization state of each forward pass that
    feeds the block, as a list of context managers. By default it follows the
    algorithm's objective (see _gpxq_pass_managers). Cached input stream k goes through
    pass k when the objective is mismatched, and the single stream goes through pass 0
    otherwise. The next block's input is propagated through the same passes.
    """
    if reset_float_cache_every is not None:
        if not isinstance(reset_float_cache_every, int) or reset_float_cache_every <= 0:
            raise ValueError("reset_float_cache_every must be None or a positive integer.")

    # Algorithms such as GPFQ and Qronos aim to solve the mismatched objective by using
    # float inputs and (possibly quantized) inputs from the previously quantized layers.
    solves_mismatched_objective = context_manager_func.solves_mismatched_objective
    if pass_managers is None:
        pass_managers = _default_pass_managers(context_manager_func, context_manager_kwargs)
    float_pass = 1 if solves_mismatched_objective else 0

    cache_state = model.config.use_cache
    model.config.use_cache = False
    blocks = recurse_getattr(model, block_name)
    first_block = blocks[0]

    # Cached inputs of the float stream (the only stream unless the objective is mismatched).
    float_cached_args, float_cached_kwargs = [], []
    # Cached inputs from the quantized model if solves_mismatched_objective is True.
    quant_cached_args, quant_cached_kwargs = [], []

    if solves_mismatched_objective:
        # Cache inputs from the quantized model first.
        hook = first_block.register_forward_pre_hook(
            partial(
                _intercept_input, cached_args=quant_cached_args, cached_kwargs=quant_cached_kwargs),
            with_kwargs=True)
        with pass_managers(model)[0]:
            for inps in dataloader:
                try:
                    model(**inps)
                except StopFwdException:
                    pass
        hook.remove()

    hook = first_block.register_forward_pre_hook(
        partial(_intercept_input, cached_args=float_cached_args, cached_kwargs=float_cached_kwargs),
        with_kwargs=True)
    with pass_managers(model)[float_pass]:
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass
    hook.remove()

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        # The passes are built per block, which is cheaper than disabling quantization for the entire model.
        block_pass_managers = pass_managers(block)
        if solves_mismatched_objective:
            passes = [(quant_cached_args, quant_cached_kwargs, block_pass_managers[0]),
                      (float_cached_args, float_cached_kwargs, block_pass_managers[1])]
        else:
            passes = [(float_cached_args, float_cached_kwargs, pass_manager)
                      for pass_manager in block_pass_managers]
        # The context manager installs hooks for the current block.
        # The callback runs the passes expected by these hooks.
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            block_optimization_callback(gpxq, passes)

        if index < len(blocks) - 1:
            # Once the block is done, we need to update the input to the next block
            if solves_mismatched_objective:
                past_quant_cached_args = deepcopy(quant_cached_args)
                quant_cached_args = []
                hook = block.register_forward_hook(
                    partial(_intercept_output, cached_args=quant_cached_args), with_kwargs=True)
                with block_pass_managers[0]:
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
            with block_pass_managers[float_pass]:
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


def _select_gpxq_class(
        base_class: Type[GPxQ], piso_class: Type[PiSOMixin],
        piso_kwargs: Optional[dict]) -> Type[GPxQ]:
    if piso_kwargs is not None:
        return partial(piso_class, **piso_kwargs)
    return base_class


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
        buffer_dtype=torch.float32,
        piso_kwargs=None):
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
    else:
        context_manager_kwargs['gptq_class'] = _select_gpxq_class(
            GPTQ, PiSOInterleavedGPTQ, piso_kwargs)
    if block_name is not None:
        _block_optimization(
            model, dataloader, block_name, context_manager_func, context_manager_kwargs)
    else:
        _calibrate_model(model, dataloader, context_manager_func, context_manager_kwargs)


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
            reset_float_cache_every=1)
    else:
        _calibrate_model(model, dataloader, context_manager_func, context_manager_kwargs)


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
    buffer_dtype=torch.float32,
    piso_kwargs=None,
):
    assert alpha > 0, "Error: alpha needs to be strictly positive"
    qronos_class = _select_gpxq_class(Qronos, PiSOInterleavedQronos, piso_kwargs)
    # Use paired quantized and float passes to correct errors from previous layers.
    _apply_gpfq_or_qronos(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=partial(qronos_class, alpha=alpha),
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
