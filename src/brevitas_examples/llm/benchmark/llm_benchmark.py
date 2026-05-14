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

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import EntryPointUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.llm_args import create_args_parser as create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args
from brevitas_examples.llm.metric_report import LLM_LOG_METRICS
from brevitas_examples.llm.metric_report import LogMetric
from brevitas_examples.llm.metric_report import result_keys


class LLMEntryPointUtils(EntryPointUtils):

    argument_parser: ArgumentParser = create_llm_args_parser()
    # Adding a metric only requires extending this tuple. Subclasses can do the
    # same with `log_metrics = LLMEntryPointUtils.log_metrics + (LogMetric(...),)`.
    log_metrics: Tuple[LogMetric, ...] = LLM_LOG_METRICS
    eval_metrics: List[str] = result_keys(log_metrics)

    @classmethod
    def parse_log(cls, job_log: str) -> Dict[str, Any]:
        job_log_results: Dict[str, Any] = {}
        for metric in cls.log_metrics:
            value, standard_error = metric.parse(job_log)
            job_log_results[metric.name] = value
            job_log_results[metric.std_name] = standard_error
        # Search for dictionary in log
        few_shot_eval_line = re.findall(r"({.*?})", job_log)
        # Retrieve last dictionary, in case other dictionaries were printed to the log
        few_shot_eval = eval(few_shot_eval_line[-1]) if len(few_shot_eval_line) > 0 else {}
        # Return the results from the log as a dictionary
        job_log_results.update(few_shot_eval)
        return job_log_results

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_llm_args(args=args, extra_args=extra_args)

    @staticmethod
    def entrypoint_main(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        from brevitas_examples.llm.main import quantize_llm
        return quantize_llm(args=args, extra_args=extra_args)


class LLMGridBenchmark(BenchmarkUtils):
    entry_point_utils = LLMEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    LLMGridBenchmark.run(sys.argv[1:])
