# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import RandomSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils


class LLMRandomBenchmark(BenchmarkUtils):
    entry_point_utils = LLMEntryPointUtils
    search_utils = RandomSearchUtils


if __name__ == "__main__":
    LLMRandomBenchmark.run(sys.argv[1:])
