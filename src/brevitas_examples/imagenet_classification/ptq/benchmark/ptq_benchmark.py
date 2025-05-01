# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import create_args_parser
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import \
    validate as validate_args


class ImagenetPTQBenchmarkUtils(BenchmarkUtils):

    argument_parser: ArgumentParser = create_args_parser()
    eval_metrics: List[str] = ["quant_top1"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        # Find the line containing Quant Top1 accuracy
        quant_top1_line = re.search(r"Total:Avg acc@1 (\d+\.\d+)", job_log)
        quant_top1 = float(quant_top1_line.group(1)) if quant_top1_line is not None else None
        # Return the results from the log as a dictionary
        job_log_results = {
            "quant_top1": quant_top1,}
        return job_log_results

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_args(args=args, extra_args=extra_args)

    @staticmethod
    def entrypoint_main(args: Namespace,
                        extra_args: Optional[List[str]] = None) -> Tuple[Dict, Any]:
        from brevitas_examples.imagenet_classification.ptq.ptq_evaluate import quantize_ptq_imagenet
        return quantize_ptq_imagenet(args=args, extra_args=extra_args)


if __name__ == "__main__":
    benchmark(ImagenetPTQBenchmarkUtils, sys.argv[1:])
