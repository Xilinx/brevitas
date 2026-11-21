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


class LLMEntryPointUtils(EntryPointUtils):

    argument_parser: ArgumentParser = create_llm_args_parser()
    eval_metrics: List[str] = [
        "float_ppl_train",
        "quant_ppl_train",
        "quant_ear_train",
        "quant_kld_train",
        "float_ppl_test",
        "quant_ppl_test",
        "quant_ear_test",
        "quant_kld_test"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        # Find train and test values in each metric line.
        float_ppl_line = re.search(
            r"Float perplexity \((.*?)\): train[:=]\s*(\d+\.\d+), test[:=]\s*(\d+\.\d+)", job_log)
        float_ppl_train = float(float_ppl_line.group(2)) if float_ppl_line is not None else None
        float_ppl_test = float(float_ppl_line.group(3)) if float_ppl_line is not None else None
        quant_ppl_line = re.search(
            r"Quantized perplexity \((.*?)\): train[:=]\s*(\d+\.\d+), test[:=]\s*(\d+\.\d+)",
            job_log)
        quant_ppl_train = float(quant_ppl_line.group(2)) if quant_ppl_line is not None else None
        quant_ppl_test = float(quant_ppl_line.group(3)) if quant_ppl_line is not None else None
        quant_ear_line = re.search(
            r"Quantized expected acceptance rate @ top-\d+ \((.*?)\): "
            r"train[:=]\s*(\d+\.\d+), test[:=]\s*(\d+\.\d+)",
            job_log)
        quant_ear_train = float(quant_ear_line.group(2)) if quant_ear_line is not None else None
        quant_ear_test = float(quant_ear_line.group(3)) if quant_ear_line is not None else None
        quant_kld_line = re.search(
            r"Quantized KL divergence @ top-\d+ \((.*?)\): "
            r"train[:=]\s*(-?\d+\.\d+), test[:=]\s*(-?\d+\.\d+)",
            job_log)
        quant_kld_train = float(quant_kld_line.group(2)) if quant_kld_line is not None else None
        quant_kld_test = float(quant_kld_line.group(3)) if quant_kld_line is not None else None
        # Search for dictionary in log
        few_shot_eval_line = re.findall(r"({.*?})", job_log)
        # Retrieve last dictionary, in case other dictionaries were printed to the log
        few_shot_eval = eval(few_shot_eval_line[-1]) if len(few_shot_eval_line) > 0 else {}
        # Return the results from the log as a dictionary
        job_log_results = {
            "float_ppl_train": float_ppl_train,
            "quant_ppl_train": quant_ppl_train,
            "quant_ear_train": quant_ear_train,
            "quant_kld_train": quant_kld_train,
            "float_ppl_test": float_ppl_test,
            "quant_ppl_test": quant_ppl_test,
            "quant_ear_test": quant_ear_test,
            "quant_kld_test": quant_kld_test,
            **few_shot_eval,}
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
