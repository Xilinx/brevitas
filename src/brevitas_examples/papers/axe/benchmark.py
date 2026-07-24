# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import sys
from typing import List
from typing import Optional

from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMGridBenchmark


class AXEEntryPointUtils(LLMEntryPointUtils):

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        LLMEntryPointUtils.validate(args=args, extra_args=extra_args)
        assert (int(args.gptq) + int(args.gpfq) + int(args.qronos)) == 1
        assert args.weight_scale_precision == args.input_scale_precision


class AXEBenchmark(LLMGridBenchmark):
    entry_point_utils = AXEEntryPointUtils


if __name__ == "__main__":
    AXEBenchmark.run(sys.argv[1:])
