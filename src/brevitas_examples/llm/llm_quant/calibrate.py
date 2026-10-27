# Copyright (C) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from functools import partial
from typing import List

from accelerate.utils.operations import send_to_device
import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.piso import optimize_scale_mode
from brevitas.graph.piso import ScaleOptimizer
from brevitas.graph.piso import ScaleOptimizerGroupInterleaved
from brevitas_examples.llm.llm_quant.gpxq import _block_optimization

# The scale sweep and H/G matrices use float32 with this tolerance.
S_DTYPE = torch.float32
S_TOLERANCE = 1e-6


@torch.no_grad()
def apply_calibration(model, dataloader):
    with calibration_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)


def build_scale_optimizer(args: Namespace) -> ScaleOptimizer:
    """Build the scale optimizer that runs with GPTQ or Qronos.

    No objective is passed: it is implied by the error-correction algorithm (GPTQ:
    self-activation, Qronos: cross-activation), whose PiSO classes decide whether
    the optimizer receives G.
    """
    if args.weight_quant_granularity == "per_group":
        ScaleOptimizerClass = (
            ScaleOptimizer
            if args.sopt_group_gpxq_layer_interleaved else ScaleOptimizerGroupInterleaved)
    elif args.weight_quant_granularity == "per_channel":
        ScaleOptimizerClass = ScaleOptimizer
    else:
        raise NotImplementedError()
    return ScaleOptimizerClass(
        solver_dtype=S_DTYPE,
        # Use the GPU for the solver when it is available.
        solver_device='cuda' if torch.cuda.is_available() else 'cpu',
        tolerance=S_TOLERANCE,
        solver_batch_size=args.sopt_solver_batch_size,
        per_group_greedy=args.sopt_group_sequential,
        per_group_greedy_reorder=args.sopt_group_sequential_act_order,
        hessian_mode=args.sopt_hessian_mode,
        solver=args.sopt_solver)


def _pass_managers(
        model: nn.Module, objective: str,
        use_quant_activations: bool) -> List[quantization_status_manager]:
    """Quantization state of each forward pass that feeds PiSO for every calibration batch.

    The objective picks the activations X_tilde (reconstruction) and X (target) the
    scale is fit to in ||X w - s X_tilde q(w)||^2:
      - unquantized: both from a pass with weight quantization off;
      - self-activation: both from the quantized model;
      - cross-activation: X_tilde from the quantized model, X from a float pass.
    Activation and bias quantization in the quantized pass follow use_quant_activations.
    The float pass of cross-activation has to come second: the layer handler pairs its
    input with the one stored by the quantized pass.
    """
    quant_pass = quantization_status_manager(
        model=model,
        disable_act_quant=not use_quant_activations,
        disable_weight_quant=objective == 'unquantized',
        disable_bias_quant=not use_quant_activations,
        is_training=False)
    if objective != 'cross-activation':
        return [quant_pass]
    float_pass = quantization_status_manager(
        model=model,
        disable_act_quant=True,
        disable_weight_quant=True,
        disable_bias_quant=True,
        is_training=False)
    return [quant_pass, float_pass]


def _piso_block_optimization_callback(
        scale_opt,
        cached_args,
        cached_kwargs,
        _unused_quant_cached_args,
        _unused_quant_cached_kwargs,
        _unused_disable_quantization_cm,
        objective,
        use_quant_activations):
    # optimize_scale_mode leaves solves_mismatched_objective False, so the block-wise
    # driver caches a single stream of block inputs. Both passes of the cross-activation
    # objective run on it: they differ only in the quantization state of the block.
    pass_managers = _pass_managers(scale_opt.model, objective, use_quant_activations)
    for _ in tqdm(range(scale_opt.num_layers), desc="Layers", leave=False):
        for args, kwargs in zip(cached_args, cached_kwargs):
            args = send_to_device(args, 'cuda')
            kwargs = send_to_device(kwargs, 'cuda')
            for pass_manager in pass_managers:
                with pass_manager:
                    scale_opt.model(*args, **kwargs)
        # The managers are exited by now: the layer's weight quantization must be
        # enabled for its scales to be written.
        scale_opt.update()


@torch.no_grad()
def apply_weight_calibration(
        model: nn.Module, dataloader: torch.utils.data.DataLoader, args: Namespace) -> None:

    objective = args.sopt_objective
    use_quant_activations = args.sopt_use_quant_activations
    context_manager_kwargs = {
        'single_forward': False,
        'solver_dtype': S_DTYPE,
        'matrix_dtype': S_DTYPE,
        'tolerance': S_TOLERANCE,
        'solver_batch_size': args.sopt_solver_batch_size,
        'cross_act_objective': objective == 'cross-activation',
        'per_group_greedy': args.sopt_group_sequential,
        'per_group_greedy_reorder': args.sopt_group_sequential_act_order,
        'hessian_mode': args.sopt_hessian_mode,
        'solver': args.sopt_solver,}

    if args.gpxq_block_name is not None:
        _block_optimization(
            model,
            dataloader,
            args.gpxq_block_name,
            optimize_scale_mode,
            context_manager_kwargs,
            block_optimization_callback=partial(
                _piso_block_optimization_callback,
                objective=objective,
                use_quant_activations=use_quant_activations),
            use_quant_activations=use_quant_activations)
        return

    pass_managers = _pass_managers(model, objective, use_quant_activations)
    # The context manager installs hooks used by PiSO. Run every pass of a batch before
    # update consumes the collected inputs.
    with optimize_scale_mode(model=model, **context_manager_kwargs) as scale_opt:
        for _ in tqdm(range(scale_opt.num_layers), desc="Layers"):
            for inps in dataloader:
                for pass_manager in pass_managers:
                    with pass_manager:
                        model(**inps)
            scale_opt.update()
