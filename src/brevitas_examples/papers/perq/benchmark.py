# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils


class PeRQBenchmark(BenchmarkUtils):
    entry_point_utils = LLMEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    PeRQBenchmark.run(sys.argv[1:])
