# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark entry point reproducing the tables of the PiSO paper.

The YAML configs sweep the scale-selection axis (``--sopt-*``) against the error
correction axis (``--gptq`` / ``--qronos``). Their Cartesian product contains many
combinations that do not correspond to a row of the paper, e.g. an ``absmax`` run
repeated once per ``--sopt-objective`` value, or a data-free run combined with a
group heuristic that it ignores. :meth:`PiSOEntryPointUtils.validate` raises
``AssertionError`` on those, which ``GridSearchUtils.gen_search_space`` uses to
discard them, so every paper row is run exactly once.
"""

from argparse import Namespace
import sys
from typing import List
from typing import Optional

from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMGridBenchmark

# PiSO minimizes the same reconstruction objective as the algorithm it is combined
# with (paper, Section 4): GPTQ uses the self-activation objective, while Qronos
# and standalone RTN use the cross-activation one.
GPTQ_OBJECTIVE = 'self-activation'
CROSS_OBJECTIVE = 'cross-activation'

# Canonical value for runs in which no scale optimization happens, so that the
# swept objective collapses to a single experiment instead of one per value.
UNUSED_OBJECTIVE = CROSS_OBJECTIVE


class PiSOEntryPointUtils(LLMEntryPointUtils):
    """Keeps only the argument combinations that map to a row of the paper."""

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        LLMEntryPointUtils.validate(args=args, extra_args=extra_args)

        if not args.sopt_optimize:
            # absmax baseline: without scale optimization every --sopt-* modifier is
            # inert, so pin them to discard the duplicates they would generate.
            assert args.sopt_hessian_mode != 'identity', \
                "The data-free baseline requires --sopt-optimize."
            assert not args.sopt_group_sequential, \
                "Group heuristics require --sopt-optimize."
            assert args.sopt_objective == UNUSED_OBJECTIVE, \
                "The objective is unused without --sopt-optimize; pinned to deduplicate."
            return

        # The --sopt-* validation only constrains the objective on the interleaved
        # path; the paper matches it to the error correction algorithm on the
        # decoupled path too.
        expected_objective = GPTQ_OBJECTIVE if args.gptq else CROSS_OBJECTIVE
        assert args.sopt_objective == expected_objective, \
            "The scale objective must match the error correction algorithm."

        if args.sopt_hessian_mode == 'identity':
            # Data-free baseline: minimizes ||w - s q(w)||^2 through the diagonal
            # path, which never reads the group heuristic.
            assert not args.sopt_group_sequential, \
                "Group heuristics are inert for the data-free baseline; pinned to deduplicate."
            # Layer-interleaved optimizes the scale before error correction starts on
            # the layer, so the weights are still the original ones and the scales
            # match the decoupled run exactly. Group-interleaved is genuinely
            # different: each group is fitted on weights that already carry the error
            # diffused by the preceding groups.
            if args.sopt_optimize_in_gpxq:
                assert not args.sopt_group_gpxq_layer_interleaved, \
                    "Layer-interleaved data-free reproduces the decoupled run; deduplicated."
                assert args.weight_quant_granularity == 'per_group', \
                    "Per-channel data-free is always layer-interleaved; deduplicated."

        if args.weight_quant_granularity == 'per_channel':
            assert not args.sopt_group_sequential, \
                "--sopt-group-sequential requires per-group weights."


class PiSOBenchmark(LLMGridBenchmark):
    entry_point_utils = PiSOEntryPointUtils


if __name__ == "__main__":
    PiSOBenchmark.run(sys.argv[1:])
