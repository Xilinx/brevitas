# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from llm_benchmark import LLMBenchmarkUtilsMixin

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import RandomSearchBenchmarkUtils


class LLMRandomSearchBenchmarkUtils(RandomSearchBenchmarkUtils, LLMBenchmarkUtilsMixin):
    pass


if __name__ == "__main__":
    benchmark(LLMRandomSearchBenchmarkUtils, sys.argv[1:])
