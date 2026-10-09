# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from typing import List
from typing import Optional
from warnings import warn

from brevitas_examples.common.parse_utils import AlgorithmArgumentParser

# Default prefix for scale optimization arguments: flags are ``--sopt-{suffix}``,
# stored as ``args.sopt_{suffix}``. PiSO is the current solver behind them.
SOPT_PREFIX = "sopt"


class ScaleOptimizerArgumentParser(AlgorithmArgumentParser):
    """Scale optimization argument group (PiSO being the current solver).

    Adds the ``--sopt-*`` flags (e.g. ``--sopt-optimize``, ``--sopt-objective``),
    read back as ``args.sopt_*``, and validates their cross-constraints.
    """

    prefix = SOPT_PREFIX

    @staticmethod
    def add_arguments(parser: ArgumentParser, prefix: str = SOPT_PREFIX) -> None:

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'solver',
            type=str,
            default='piso',
            choices=['piso', 'chunked-piso'],
            help='Scale optimization solver family. chunked-piso is a faster '
            'per-channel variant that only supports positive scales (symmetric '
            'grids). Default: piso')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'optimize',
            default=False,
            help='Apply scale optimization based on the calibration set. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'optimize-in-gpxq',
            default=False,
            help='Apply scale optimization together with gpxq or qronos. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'use-quant-activations',
            default=True,
            help=f'With --{prefix}-optimize, use quant activations when optimising the scale. '
            'Default: enabled')

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'solver-batch-size',
            type=int,
            default=None,
            help=f'With --{prefix}-optimize, the number of weight rows/channels in a given layer '
            'for which the scale is optimized in parallel. When set to None all the scales in the '
            'layer are optimized at the same time (fastest but more memory required).')

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'objective',
            type=str,
            default='unquantized',
            choices=['cross-activation', 'self-activation', 'unquantized'],
            help=f'With --{prefix}-optimize, the reconstruction objective the solver optimizes, '
            'i.e. the choice of X (target) and X_tilde (reconstruction) in '
            '||X w - s X_tilde q(w)||^2. '
            'cross-activation: X float, X_tilde quantized (Qronos/GPTAQ target). '
            'self-activation: both X_tilde quantized (GPTQ target). '
            'unquantized: both float X (data-aware weight MSE). Default: unquantized')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential',
            default=False,
            help=f'With --{prefix}-optimize and per-group weights, optimize each group correcting '
            'the error from previous groups (sequential heuristic). When disabled, groups are '
            'optimized independently. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential-act-order',
            default=True,
            help=f'With --{prefix}-group-sequential (standalone --{prefix}-optimize only), process '
            'groups by descending sum of diag(H) (most important first) instead of natural order. '
            f'Ignored with --{prefix}-optimize-in-gpxq, where the column order is controlled by '
            '--gpxq-act-order. Default: enabled')

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'hessian-mode',
            type=str,
            default='dense',
            choices=['dense', 'diagonal', 'identity'],
            help=f'With --{prefix}-optimize, how the Hessian H (and G) is used by the solver. '
            'dense: the full matrices accumulated from the calibration data; not implemented, so '
            'it currently raises NotImplementedError. '
            'diagonal: only diag(H) (and diag(G)), discarding cross-weight interactions. '
            'identity: ignore the calibration data entirely and use H=I, G=None (pure '
            'weight MSE). Default: dense')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'group-gpxq-layer-interleaved',
            default=False,
            help='Selects how the solver is interleaved with GPxQ for group-wise weight '
            'quantization (paper: interleaved integration strategies). When enabled, all group '
            'scales are optimized once per layer (layer-interleaved). When disabled (default), '
            'each group scale is optimized at its boundary during GPxQ (group-interleaved). Only '
            f'applies with --{prefix}-optimize-in-gpxq and per-group weights; per-channel '
            'quantization is always layer-interleaved regardless. Default: disabled')

    @staticmethod
    def validate(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            prefix: str = SOPT_PREFIX) -> None:
        # Read attributes through the prefix so checks stay prefix-configurable.
        def a(suffix: str):
            return getattr(args, ScaleOptimizerArgumentParser._dest(prefix, suffix))

        optimize = a('optimize')
        optimize_in_gpxq = a('optimize-in-gpxq')
        objective = a('objective')
        layer_interleaved = a('group-gpxq-layer-interleaved')
        hessian_mode = a('hessian-mode')

        # Scale optimization: --{prefix}-optimize is the master switch,
        # --{prefix}-optimize-in-gpxq is a modifier selecting the interleaved strategy.
        if optimize_in_gpxq:
            assert optimize, \
                f"--{prefix}-optimize-in-gpxq requires --{prefix}-optimize (it selects the interleaved strategy)."
            assert args.gptq or args.qronos, \
                f"--{prefix}-optimize-in-gpxq requires --gptq or --qronos (the solver is interleaved into them)."
            # When interleaved, the input quantization is dictated by the error-correction
            # algorithm, so the scale objective must match it: GPTQ uses the self-activation
            # objective, Qronos uses the cross-activation one.
            if args.gptq:
                assert objective == 'self-activation', \
                    f"--{prefix}-optimize-in-gpxq with --gptq requires --{prefix}-objective self-activation."
            if args.qronos:
                assert objective == 'cross-activation', \
                    f"--{prefix}-optimize-in-gpxq with --qronos requires --{prefix}-objective cross-activation."
            # The group- vs layer-interleaved choice only exists for per-group weights;
            # per-channel is always layer-interleaved.
            if layer_interleaved:
                assert args.weight_quant_granularity == 'per_group', \
                    f"--{prefix}-group-gpxq-layer-interleaved only applies to per-group weight quantization."
        else:
            assert not layer_interleaved, \
                f"--{prefix}-group-gpxq-layer-interleaved requires --{prefix}-optimize-in-gpxq."

        if optimize or optimize_in_gpxq:
            if hessian_mode == 'diagonal':
                warn(
                    f"--{prefix}-hessian-mode diagonal: only the diagonal of H (and G) is used for "
                    "scale optimization; off-diagonal cross-weight interactions are discarded.")
            elif hessian_mode == 'identity':
                warn(
                    f"--{prefix}-hessian-mode identity: scale optimization is data-free (H=I, "
                    "G=None, pure weight MSE). Note the calibration forward passes still run with "
                    "the current implementation, but the computed H and G matrices are ignored.")
