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
from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.stable_diffusion.stable_diffusion_args import create_args_parser
from brevitas_examples.stable_diffusion.stable_diffusion_args import validate as validate_args


class SDBenchmarkUtils(BenchmarkUtils):

    argument_parser: ArgumentParser = create_args_parser()
    eval_metrics: List[str] = ["torchmetrics_fid", "clean_fid"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        torchmetrics_fid_line = re.search(r"Torchmetrics FID: (\d+\.\d+)", job_log)
        torchmetrics_fid = float(
            torchmetrics_fid_line.group(1)) if torchmetrics_fid_line is not None else None

        clean_fid_line = re.search(r"Cleanfid FID: (\d+\.\d+)", job_log)
        clean_fid = float(clean_fid_line.group(1)) if clean_fid_line is not None else None
        # Return the results from the log as a dictionary
        job_log_results = {
            "torchmetrics_fid": torchmetrics_fid,
            "clean_fid": clean_fid,}
        return job_log_results

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_args(args=args, extra_args=extra_args)

    @staticmethod
    def entrypoint_main(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        # Override output_path to ensure that all the experiment results are saved
        # in job_folder
        args.output_path = job_folder
        from brevitas_examples.stable_diffusion.main import quantize_sd
        return quantize_sd(args=args, extra_args=extra_args)


if __name__ == "__main__":
    benchmark(SDBenchmarkUtils, sys.argv[1:])
