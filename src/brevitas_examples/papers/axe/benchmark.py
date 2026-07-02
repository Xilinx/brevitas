# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.llm.benchmark.llm_benchmark import LLMBenchmarkUtils


class AXEBenchmark(LLMBenchmarkUtils):

    @staticmethod
    def validate(args, extra_args=None):
        super(LLMBenchmarkUtils, AXEBenchmark).validate(args, extra_args)
        assert (int(args.gptq) + int(args.gpfq) + int(args.qronos)) == 1
        assert args.weight_scale_precision == args.input_scale_precision


if __name__ == "__main__":
    benchmark(AXEBenchmark, sys.argv[1:])
