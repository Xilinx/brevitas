# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.stable_diffusion.stable_diffusion_args import create_sd_args_parser
from brevitas_examples.stable_diffusion.stable_diffusion_args import validate as validate_sd_args


class SDBenchmarkUtils(BenchmarkUtils):

    argument_parser: ArgumentParser = create_sd_args_parser()

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Log parsing for stable diffusion entrypoint is not implemented yet.")

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_sd_args(args=args, extra_args=extra_args)

    @staticmethod
    def entrypoint_main(args: Namespace,
                        extra_args: Optional[List[str]] = None) -> Tuple[Dict, Any]:
        from brevitas_examples.stable_diffusion.main import quantize_sd
        return quantize_sd(args=args, extra_args=extra_args)


if __name__ == "__main__":
    benchmark(SDBenchmarkUtils, sys.argv[1:])
