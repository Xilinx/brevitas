# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import RandomSearchMixin
from brevitas_examples.llm.benchmark.llm_benchmark import LLMBenchmarkUtilsBase


class LLMRandomSearchBenchmarkUtils(LLMBenchmarkUtilsBase, RandomSearchMixin):
    pass


if __name__ == "__main__":
    benchmark(LLMRandomSearchBenchmarkUtils, sys.argv[1:])
