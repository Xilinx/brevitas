# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from typing import List
from typing import Optional
from warnings import warn

from brevitas_examples.common.parse_utils import AlgorithmArgumentParser

SOPT_PREFIX = "sopt"


class ScaleOptimizerArgumentParser(AlgorithmArgumentParser):
    """Add and validate the scale optimization arguments."""

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
            'use-quant-activations',
            default=True,
            help='Use quantized activations during weight calibration. '
            'Default: enabled')

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'solver-batch-size',
            type=int,
            default=None,
            help='The number of weight rows/channels in a given layer '
            'for which the scale is optimized in parallel. When set to None all the scales in the '
            'layer are optimized at the same time (fastest but more memory required).')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'quantize-prev',
            default=False,
            help='Use quantized activations to reconstruct the current layer. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'cross-act-objective',
            default=False,
            help='Use float activations as the reconstruction target. '
            'Requires --sopt-quantize-prev. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential',
            default=False,
            help='With per-group weights, optimize each group while correcting '
            'the error from previous groups (sequential heuristic). When disabled, groups are '
            'optimized independently. Default: disabled')

        ScaleOptimizerArgumentParser._add_bool(
            parser,
            prefix,
            'group-sequential-act-order',
            default=True,
            help=f'With --{prefix}-group-sequential during weight calibration, process '
            'groups by descending sum of diag(H) (most important first) instead of natural order. '
            'Ignored with --gpxq-use-grid-optimization, where the column order is controlled by '
            '--gpxq-act-order. Default: enabled')

        ScaleOptimizerArgumentParser._add(
            parser,
            prefix,
            'hessian-mode',
            type=str,
            default='dense',
            choices=['dense', 'diagonal', 'identity'],
            help='How the solver uses the Hessian H and matrix G. '
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
            'applies with --gpxq-use-grid-optimization and per-group weights; per-channel '
            'quantization is always layer-interleaved regardless. Default: disabled')

    @staticmethod
    def validate(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            prefix: str = SOPT_PREFIX) -> None:

        def a(suffix: str):
            return getattr(args, ScaleOptimizerArgumentParser._dest(prefix, suffix))

        weight_calibration = args.weight_calibration
        gpxq_use_grid_optimization = args.gpxq_use_grid_optimization
        quantize_prev = a('quantize-prev')
        cross_act_objective = a('cross-act-objective')
        layer_interleaved = a('group-gpxq-layer-interleaved')
        hessian_mode = a('hessian-mode')

        assert quantize_prev or not cross_act_objective, \
            f"--{prefix}-cross-act-objective requires --{prefix}-quantize-prev."

        if gpxq_use_grid_optimization:
            assert args.gptq or args.qronos, \
                "--gpxq-use-grid-optimization requires --gptq or --qronos."
            if args.gptq:
                assert quantize_prev and not cross_act_objective, \
                    f"--gpxq-use-grid-optimization with --gptq requires --{prefix}-quantize-prev and --no-{prefix}-cross-act-objective."
            if args.qronos:
                assert quantize_prev and cross_act_objective, \
                    f"--gpxq-use-grid-optimization with --qronos requires --{prefix}-quantize-prev and --{prefix}-cross-act-objective."
            if layer_interleaved:
                assert args.weight_quant_granularity == 'per_group', \
                    f"--{prefix}-group-gpxq-layer-interleaved only applies to per-group weight quantization."
        else:
            assert not layer_interleaved, \
                f"--{prefix}-group-gpxq-layer-interleaved requires --gpxq-use-grid-optimization."

        if weight_calibration or gpxq_use_grid_optimization:
            if hessian_mode == 'diagonal':
                warn(
                    f"--{prefix}-hessian-mode diagonal: only the diagonal of H (and G) is used for "
                    "scale optimization; off-diagonal cross-weight interactions are discarded.")
            elif hessian_mode == 'identity':
                warn(
                    f"--{prefix}-hessian-mode identity: scale optimization is data-free (H=I, "
                    "G=None, pure weight MSE). Note the calibration forward passes still run with "
                    "the current implementation, but the computed H and G matrices are ignored.")
