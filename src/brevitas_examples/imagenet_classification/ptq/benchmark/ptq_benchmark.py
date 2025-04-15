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
        # TODO: Implement log parsing to extract quality metrics from entrypoint output
        raise NotImplementedError("Log parsing for Imagenet PTQ entrypoint is not implemented yet.")

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
