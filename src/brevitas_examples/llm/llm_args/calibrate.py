# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from typing import List
from typing import Optional
from warnings import warn

from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas_examples.common.parse_utils import AlgorithmArgumentParser

GRID_OPT_PREFIX = "grid-opt"
# Reconstruction objectives of --grid-optimization, see apply_grid_optimization.
GRID_OPT_OBJECTIVES = ['unquantized', 'quantized', 'mismatched']


class GridOptimizationArgumentParser(AlgorithmArgumentParser):
    """Add and validate the grid optimization arguments."""

    prefix = GRID_OPT_PREFIX

    @staticmethod
    def add_arguments(parser: ArgumentParser, prefix: str = GRID_OPT_PREFIX) -> None:
        """Add the grid optimization flags.

        The solver choices are the solvers registered in SCALE_SOLVER_REGISTRY when
        this is called, so a solver defined elsewhere is offered once the module that
        registers it has been imported.
        """
        GridOptimizationArgumentParser._add(
            parser,
            prefix,
            'solver',
            type=str,
            default='piso',
            choices=sorted(SCALE_SOLVER_REGISTRY.get_registered_keys()),
            help='Solver that optimizes the scale. piso is PiSO (Piecewise Scale '
            'Optimization); chunked-piso is a faster variant that only supports positive '
            'scales. Default: piso')

        GridOptimizationArgumentParser._add_bool(
            parser,
            prefix,
            'use-quant-activations',
            default=True,
            help='Use quantized activations with --grid-optimization. '
            'Default: enabled')

        GridOptimizationArgumentParser._add(
            parser,
            prefix,
            'solver-batch-size',
            type=int,
            default=None,
            help='The number of weight rows/channels in a given layer '
            'for which the scale is optimized in parallel. When set to None all the scales in the '
            'layer are optimized at the same time (fastest but more memory required).')

        GridOptimizationArgumentParser._add(
            parser,
            prefix,
            'objective',
            type=str,
            default='unquantized',
            choices=GRID_OPT_OBJECTIVES,
            help='Reconstruction objective of --grid-optimization, ||X w - s X_tilde q(w)||^2, '
            'where X_tilde are the activations the layer sees. '
            'unquantized: X_tilde = X, with the previous layers left unquantized. '
            'quantized: X_tilde = X, from the quantized previous layers. '
            'mismatched: X_tilde from the quantized previous layers, X from the float model. '
            'With --gpxq-use-grid-optimization the objective follows the error-correction '
            'algorithm (GPTQ: quantized, Qronos: mismatched). Default: unquantized')

        GridOptimizationArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential',
            default=False,
            help=f'With per-group weights and --{prefix}-covariance-mode dense, optimize the '
            'groups one at a time, each against the reconstruction of the groups before it. '
            'Otherwise, every group is optimized independently. Not supported with '
            'group-interleaved --gpxq-use-grid-optimization. Default: disabled')

        GridOptimizationArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential-act-order',
            default=True,
            help=f'With --{prefix}-group-sequential and --grid-optimization, process '
            'groups by descending sum of diag(H) (most important first) instead of natural order. '
            'Ignored with --gpxq-use-grid-optimization, where the column order is controlled by '
            '--gpxq-act-order. Default: enabled')

        GridOptimizationArgumentParser._add(
            parser,
            prefix,
            'covariance-mode',
            type=str,
            default='diagonal',
            choices=['dense', 'diagonal', 'identity'],
            help='How the solver uses the covariance matrices H and G. '
            'dense: the full matrices accumulated from the calibration data, for a solver that '
            'supports them (none of the current ones does); each group keeps the block of its '
            'own weights. '
            'diagonal: only diag(H) (and diag(G)), discarding cross-weight interactions. '
            'identity: ignore the calibration data entirely and use H=I, G=None (pure '
            'weight MSE); the calibration forward passes still run. Default: diagonal')

        GridOptimizationArgumentParser._add_bool(
            parser,
            prefix,
            'group-gpxq-layer-interleaved',
            default=False,
            help='Selects how the solver is interleaved with GPxQ for group-wise weight '
            'quantization (paper: interleaved integration strategies). When enabled, all group '
            'scales are optimized once per layer (layer-interleaved). When disabled (default), '
            'each group scale is optimized at its boundary during GPxQ (group-interleaved). Only '
            'applies with --gpxq-use-grid-optimization and per-group weights; per-channel '
            'quantization is always layer-interleaved regardless. Default: disabled')

    @staticmethod
    def validate(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            prefix: str = GRID_OPT_PREFIX) -> None:

        def a(suffix: str):
            return getattr(args, GridOptimizationArgumentParser._dest(prefix, suffix))

        grid_optimization = args.grid_optimization
        gpxq_use_grid_optimization = args.gpxq_use_grid_optimization
        objective = a('objective')
        layer_interleaved = a('group-gpxq-layer-interleaved')
        covariance_mode = a('covariance-mode')
        group_sequential = a('group-sequential')

        # Arguments built without argparse (e.g., benchmark sweeps) skip `choices`.
        assert objective in GRID_OPT_OBJECTIVES, \
            f"--{prefix}-objective must be one of {GRID_OPT_OBJECTIVES}, got {objective!r}."

        if gpxq_use_grid_optimization:
            assert args.gptq or args.qronos, \
                "--gpxq-use-grid-optimization requires --gptq or --qronos."
            assert args.gpxq_max_accumulator_bit_width is None, \
                "--gpxq-use-grid-optimization is not supported with --gpxq-max-accumulator-bit-width."
            if layer_interleaved:
                assert args.weight_quant_granularity == 'per_group', \
                    f"--{prefix}-group-gpxq-layer-interleaved only applies to per-group weight quantization."
            elif args.weight_quant_granularity == 'per_group':
                assert not group_sequential, \
                    f"--{prefix}-group-sequential is not supported with group-interleaved " \
                    "--gpxq-use-grid-optimization; add " \
                    f"--{prefix}-group-gpxq-layer-interleaved to combine them."
            if not grid_optimization:
                # The interleaved objective follows the algorithm, so an explicit
                # objective that disagrees with it would otherwise be ignored silently.
                implied_objective = 'quantized' if args.gptq else 'mismatched'
                if objective not in ('unquantized', implied_objective):
                    warn(
                        f"--{prefix}-objective {objective} is ignored with "
                        f"--gpxq-use-grid-optimization: the objective follows the "
                        f"error-correction algorithm ({implied_objective}).")
        else:
            assert not layer_interleaved, \
                f"--{prefix}-group-gpxq-layer-interleaved requires --gpxq-use-grid-optimization."

        if grid_optimization or gpxq_use_grid_optimization:
            solver = a('solver')
            supports_dense = SCALE_SOLVER_REGISTRY.get(solver).supports_dense
            assert covariance_mode != 'dense' or supports_dense, \
                f"--{prefix}-solver {solver} only uses the diagonal of the covariance " \
                f"matrix; use --{prefix}-covariance-mode diagonal or identity."
