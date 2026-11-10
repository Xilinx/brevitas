# Copyright (C) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from functools import partial
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.grid_optimization import grid_optimization_mode
from brevitas_examples.llm.llm_quant.gpxq import _block_optimization
from brevitas_examples.llm.llm_quant.gpxq import _calibrate_model

# The scale sweep and H/G matrices use float32 with this tolerance.
S_DTYPE = torch.float32
S_TOLERANCE = 1e-6


@torch.no_grad()
def apply_calibration(model, dataloader):
    with calibration_mode(model):
        for inps in tqdm(dataloader):
            model(**inps)


def build_grid_optimization_kwargs(args: Namespace) -> dict:
    """Arguments of InterleavedGridOptimizer.for_gpxq, which GPTQ / Qronos call for each layer.

    No objective is passed: it is implied by the error-correction algorithm (GPTQ:
    quantized, Qronos: mismatched), which decides whether the grid optimizer
    receives G.
    """
    if args.weight_quant_granularity not in ("per_group", "per_channel"):
        raise NotImplementedError(
            "Grid optimization inside GPTQ/Qronos supports per-channel and per-group "
            f"weight quantization, got {args.weight_quant_granularity}.")
    # Use the GPU for the solver when it is available.
    solver_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        'layer_interleaved': args.grid_opt_group_gpxq_layer_interleaved,
        'solver_dtype': S_DTYPE,
        'solver_device': solver_device,
        'tolerance': S_TOLERANCE,
        'solver_batch_size': args.grid_opt_solver_batch_size,
        'per_group_greedy': args.grid_opt_group_sequential,
        'per_group_greedy_reorder': args.grid_opt_group_sequential_act_order,
        'covariance_mode': args.grid_opt_covariance_mode,
        'solver': args.grid_opt_solver,}


def _pass_managers(model: nn.Module, objective: str,
                   use_quant_activations: bool) -> List[quantization_status_manager]:
    """Which quantizers are enabled in each forward pass that feeds grid optimization,
    per batch, as a list of context managers.

    The objective picks the activations X_tilde (reconstruction) and X (target) the
    scale is fit to in ||X w - s X_tilde q(w)||^2:
      - unquantized: both from a pass with weight quantization off;
      - quantized: both from the quantized model;
      - mismatched: X_tilde from the quantized model, X from a float pass.
    Activation and bias quantization in the quantized pass follow use_quant_activations.
    The float pass of the mismatched objective comes second: the grid optimization hook
    pairs its input with the one stored by the quantized pass.
    """
    quant_pass = quantization_status_manager(
        model=model,
        disable_act_quant=not use_quant_activations,
        disable_weight_quant=objective == 'unquantized',
        disable_bias_quant=not use_quant_activations,
        is_training=False)
    if objective != 'mismatched':
        return [quant_pass]
    float_pass = quantization_status_manager(
        model=model,
        disable_act_quant=True,
        disable_weight_quant=True,
        disable_bias_quant=True,
        is_training=False)
    return [quant_pass, float_pass]


@torch.no_grad()
def apply_grid_optimization(
        model: nn.Module, dataloader: torch.utils.data.DataLoader, args: Namespace) -> None:

    objective = args.grid_opt_objective
    pass_managers = partial(
        _pass_managers,
        objective=objective,
        use_quant_activations=args.grid_opt_use_quant_activations)
    context_manager_kwargs = {
        'solver_dtype': S_DTYPE,
        'matrix_dtype': S_DTYPE,
        'tolerance': S_TOLERANCE,
        'solver_batch_size': args.grid_opt_solver_batch_size,
        'solves_mismatched_objective': objective == 'mismatched',
        'per_group_greedy': args.grid_opt_group_sequential,
        'per_group_greedy_reorder': args.grid_opt_group_sequential_act_order,
        'covariance_mode': args.grid_opt_covariance_mode,
        'solver': args.grid_opt_solver,}

    if args.gpxq_block_name is not None:
        _block_optimization(
            model,
            dataloader,
            args.gpxq_block_name,
            grid_optimization_mode,
            context_manager_kwargs,
            # As for GPFQ / Qronos, the cached float inputs are reset to the quantized
            # ones after every block.
            reset_float_cache_every=1,
            pass_managers=pass_managers)
    else:
        _calibrate_model(
            model,
            dataloader,
            grid_optimization_mode,
            context_manager_kwargs,
            pass_managers=pass_managers)
