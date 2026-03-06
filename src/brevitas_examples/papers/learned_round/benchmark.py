# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.llm.benchmark.llm_benchmark import LLMBenchmarkUtils


class LearnedRoundBenchmark(LLMBenchmarkUtils):

    @staticmethod
    def validate(args, extra_args=None):
        if args.rotation == 'fused_no_fx':
            assert not args.convert_layernorm_to_rmsnorm, 'LayerNorm is automatically replaced with RMSNorm when running with --rotation=fused_no_fx. Remove the flag --convert-layernorm-to-rmsnorm'
            # Automatically replace `LayerNorm` with `RMSNorm` when running with `--rotation=fused_no_fx`
            args.replace_rmsnorm = True
            assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
        assert (int(args.magr) + int(args.rotation is not None) + int(args.act_equalization is not None)) <= 1, "Only a single preprocessing setting can be used."
        if not args.no_quantize:
            # Skip unnecessary experiments
            assert (int(args.gptq) + int(args.gpfq) + int(args.qronos)) <= 1, "GPTQ, GPFQ, and/or Qronos cannot be enabled together."
            assert args.learned_round is None or (args.learned_round is not None and (int(args.gptq) + int(args.gpfq) + int(args.qronos)) == 0), "Learned Round cannot be combined with GPTQ/GPFQ/Qronos."
        if args.weight_quant_granularity == 'per_channel':
            assert args.weight_bit_width == 4, "When per_channel only bit_width=4 is allowed"


if __name__ == "__main__":
    benchmark(LearnedRoundBenchmark, sys.argv[1:])
