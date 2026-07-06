# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.llm.benchmark.llm_benchmark import LLMBenchmarkUtils


class PeRQBenchmark(LLMBenchmarkUtils):
    pass


if __name__ == "__main__":
    benchmark(PeRQBenchmark, sys.argv[1:])
