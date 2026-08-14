# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import torch

# Increase the dynamo recompilation limit: the learned-float quantizer and the
# custom trainer trigger many recompilations during fine-tuning, and the default
# limit is too low.
torch._dynamo.config.recompile_limit = 1000

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
# Importing these modules registers the custom trainer into TRAINER_REGISTRY and
# the learned-float quantizers into QUANTIZERS_REGISTRY as an import side effect.
# This lets the benchmark YAML refer to them by bare name (e.g.
# ``rotation_learned_bitwidth`` / ``learned_float``) instead of the full
# ``path/to/plugin.py:name`` plugin path.
import brevitas_examples.papers.dMX.custom_trainer  # noqa: F401
import brevitas_examples.papers.dMX.learned_float_quantizer  # noqa: F401


class DMXEntryPointUtils(LLMEntryPointUtils):

    @staticmethod
    def entrypoint_main(args, extra_args=None, job_folder=None):
        # Benchmark workers use multiprocessing spawn, so repeat plugin
        # registration in the child process before resolving bare names.
        import brevitas_examples.papers.dMX.custom_trainer  # noqa: F401
        import brevitas_examples.papers.dMX.learned_float_quantizer  # noqa: F401

        return LLMEntryPointUtils.entrypoint_main(args, extra_args, job_folder)


class DMXBenchmark(BenchmarkUtils):
    entry_point_utils = DMXEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    DMXBenchmark.run(sys.argv[1:])
