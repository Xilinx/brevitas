# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
import re
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import RandomSearchBenchmarkUtils
from brevitas_examples.llm.llm_args import create_args_parser as create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args

from llm_benchmark import LLMBenchmarkUtilsMixin

class LLMRandomSearchBenchmarkUtils(RandomSearchBenchmarkUtils, LLMBenchmarkUtilsMixin):
    pass


if __name__ == "__main__":
    benchmark(LLMRandomSearchBenchmarkUtils, sys.argv[1:])
