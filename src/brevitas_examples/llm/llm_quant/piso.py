# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from typing import Tuple
from warnings import warn

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.piso import optimize_scale_mode
from brevitas.graph.piso import ScaleOptimizer
from brevitas.graph.piso import ScaleOptimizerGroupInterleaved
from brevitas_examples.llm.llm_quant.gpxq import block_optimization

# The scale sweep and the H/G matrices run in float32, with this interval
# tolerance. Exposed here as constants rather than CLI flags.
S_DTYPE = torch.float32
S_TOLERANCE = 1e-6


def objective_flags(s_objective: str) -> Tuple[bool, bool]:
    # Translate the user-facing --s-objective choice into the (cross_act_objective,
    # quantize_prev) pair the PiSO classes expect. These pick X (target) and
    # X_tilde (reconstruction) in ||X w - s X_tilde q(w)||^2:
    #   cross-activation -> X float, X_tilde quantized (needs quantized prev layer)
    #   self-activation  -> both X_tilde quantized
    #   unquantized      -> both float X
    return {
        'cross-activation': (True, True),
        'self-activation': (False, True),
        'unquantized': (False, False),}[s_objective]


def build_scale_optimizer(args: Namespace, cross_act_objective: bool) -> ScaleOptimizer:
    # Build the ScaleOptimizer used to interleave PiSO with GPTQ/Qronos
    # (the --sopt-optimize-in-gpxq path). Per-group quantization uses the group-wise
    # (group-interleaved) updater unless --sopt-group-gpxq-layer-interleaved selects
    # the layer-interleaved one; per-channel is always layer-interleaved.
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
        # TODO (pml): Consider generalizing this logic
        solver_device='cuda' if torch.cuda.is_available() else 'cpu',
        tolerance=S_TOLERANCE,
        solver_batch_size=args.sopt_solver_batch_size,
        cross_act_objective=cross_act_objective,
        per_group_greedy=args.sopt_group_sequential,
        per_group_greedy_reorder=args.sopt_group_sequential_act_order,
        hessian_mode=args.sopt_hessian_mode,
        solver=args.sopt_solver)


@torch.no_grad()
def apply_scale_optimization(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        args: Namespace,  # TODO (jpga) add option for negative scale whenever that works in brevitas.
) -> None:
    """Run standalone (decoupled) PiSO scale optimization over ``model``.

    This is the entry point for optimizing scales on their own, before/without error
    correction (the ``--sopt-optimize`` path). The ``--sopt-*`` settings are read
    from ``args``. It drives ``optimize_scale_mode`` over the calibration
    ``dataloader``: block-by-block if ``--gpxq-block-name`` is given, otherwise
    across the whole model. To interleave PiSO with GPTQ/Qronos instead, build a
    ``ScaleOptimizer`` and pass it into ``apply_gptq`` / ``apply_qronos``.
    """
    if args.input_bit_width is not None:
        warn(
            "Scale optimization with activation (input) quantization is untested; "
            "results may be unreliable.")

    cross_act_objective, quantize_prev = objective_flags(args.sopt_objective)
    # single_forward and the dtypes/tolerance are not user-facing; use the module
    # constants (see S_DTYPE / S_TOLERANCE).
    single_forward = False
    context_manager_kwargs = {
        'use_quant_activations': args.sopt_use_quant_activations,
        'quantize_prev': quantize_prev,
        'single_forward': single_forward,
        'solver_dtype': S_DTYPE,
        'matrix_dtype': S_DTYPE,
        'tolerance': S_TOLERANCE,
        'solver_batch_size': args.sopt_solver_batch_size,
        'cross_act_objective': cross_act_objective,
        'per_group_greedy': args.sopt_group_sequential,
        'per_group_greedy_reorder': args.sopt_group_sequential_act_order,
        'hessian_mode': args.sopt_hessian_mode,
        'solver': args.sopt_solver,}

    if args.gpxq_block_name is not None:
        block_optimization(
            model, dataloader, args.gpxq_block_name, optimize_scale_mode, context_manager_kwargs)
        return

    with torch.no_grad():
        with optimize_scale_mode(model=model, **context_manager_kwargs) as scale_opt:
            num_iter = scale_opt.num_layers if not single_forward else 1
            for _ in tqdm(range(num_iter), desc="Layers"):
                for inps in dataloader:
                    model(**inps)
                scale_opt.update()
