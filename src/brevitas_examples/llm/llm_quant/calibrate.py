# Copyright (C) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.calibrate import calibration_mode
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
    """Build the scale optimizer that runs with GPTQ or Qronos."""
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
        cross_act_objective=args.sopt_cross_act_objective,
        per_group_greedy=args.sopt_group_sequential,
        per_group_greedy_reorder=args.sopt_group_sequential_act_order,
        hessian_mode=args.sopt_hessian_mode,
        solver=args.sopt_solver)


@torch.no_grad()
def apply_weight_calibration(
        model: nn.Module, dataloader: torch.utils.data.DataLoader, args: Namespace) -> None:

    context_manager_kwargs = {
        'use_quant_activations': args.sopt_use_quant_activations,
        'quantize_prev': args.sopt_quantize_prev,
        'single_forward': False,
        'solver_dtype': S_DTYPE,
        'matrix_dtype': S_DTYPE,
        'tolerance': S_TOLERANCE,
        'solver_batch_size': args.sopt_solver_batch_size,
        'cross_act_objective': args.sopt_cross_act_objective,
        'per_group_greedy': args.sopt_group_sequential,
        'per_group_greedy_reorder': args.sopt_group_sequential_act_order,
        'hessian_mode': args.sopt_hessian_mode,
        'solver': args.sopt_solver,}

    if args.gpxq_block_name is not None:
        _block_optimization(
            model, dataloader, args.gpxq_block_name, optimize_scale_mode, context_manager_kwargs)
        return

    with optimize_scale_mode(model=model, **context_manager_kwargs) as scale_opt:
        for _ in tqdm(range(scale_opt.num_layers), desc="Layers"):
            for inps in dataloader:
                model(**inps)
            scale_opt.update()
