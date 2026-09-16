# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""FSDP-aware benchmark entrypoint for LLM quantization with fine-tuning.

Unlike :class:`LLMEntryPointUtils` (which calls ``quantize_llm`` directly
in-process), this entrypoint launches ``main.py`` under ``accelerate launch``
so that ``torch.distributed`` is properly initialized and FSDP can shard
the model across all available GPUs.

``accelerate launch`` reads its FSDP2 sharding, wrapping, and mixed-precision
settings from the config file provided with ``--accelerate-config``.

The number of processes is derived from ``CUDA_VISIBLE_DEVICES`` (already set
by the benchmark framework for each worker).  Rank 0 writes results to
``job_folder/results.json``; the parent reads that file after the subprocess
exits.

Usage::

    python -m brevitas_examples.llm.benchmark.llm_fsdp_benchmark \\
        --config benchmark_template.yaml \\
        --results-folder ./results \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --num-gpus-per-process 8 \\
        --accelerate-config path/to/accelerate_config.yaml  # optional \\
        --quiet                                              # optional
"""

from argparse import ArgumentParser
from argparse import Namespace
import json
import os
import socket
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import yaml

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import EntryPointUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
from brevitas_examples.llm.llm_args import create_args_parser as create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _resolve_llm_config(args: Namespace) -> Dict[str, Any]:
    config = {}
    if args.config is not None:
        with open(args.config) as config_file:
            config = yaml.safe_load(config_file) or {}

    defaults = {
        action.dest: action.default for action in LLMFSDPEntryPointUtils.argument_parser._actions}
    explicit_keys = getattr(args, "_benchmark_explicit_keys", set())
    for key, value in vars(args).items():
        if key in {"config", "job_folder", "_benchmark_explicit_keys"}:
            continue
        if key in explicit_keys or key not in config or value != defaults.get(key):
            config[key] = value
    return config


def _validate_accelerate_config(config_path: str) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Accelerate config not found: {config_path}")
    with open(config_path) as config_file:
        accelerate_config = yaml.safe_load(config_file) or {}
    if str(accelerate_config.get("distributed_type", "")).upper() != "FSDP":
        raise ValueError("The Accelerate config must set distributed_type to FSDP.")
    fsdp_config = accelerate_config.get("fsdp_config", {})
    if int(fsdp_config.get("fsdp_version", 1)) != 2:
        raise ValueError("The Accelerate config must set fsdp_config.fsdp_version to 2.")


class LLMFSDPEntryPointUtils(EntryPointUtils):
    """Benchmark utilities that launch ``main.py`` under ``accelerate launch``.

    This is intended for FSDP fine-tuning jobs where
    ``torch.distributed`` must be initialized before ``quantize_llm``
    runs. Non-FSDP jobs should use :class:`LLMEntryPointUtils` instead.
    """

    argument_parser: ArgumentParser = create_llm_args_parser()
    eval_metrics: List[str] = ["float_ppl", "quant_ppl"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        return LLMEntryPointUtils.parse_log(job_log)

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        effective_args = Namespace(**vars(args))
        for key, value in _resolve_llm_config(args).items():
            if hasattr(effective_args, key):
                setattr(effective_args, key, value)
        validate_llm_args(args=effective_args, extra_args=extra_args)
        assert effective_args.fine_tune, "The FSDP2 benchmark requires --fine-tune."

    @staticmethod
    def entrypoint_main(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        """Launch ``main.py`` under ``accelerate launch`` and collect results.

        Parameters
        ----------
        args : Namespace
            Parsed LLM quantization arguments.
        extra_args : list of str, optional
            Additional CLI tokens (e.g. HuggingFace ``TrainingArguments``).
        job_folder : str, optional
            Directory where rank 0 writes ``results.json``.  Required for
            result collection.

        Returns
        -------
        tuple of (dict, None)
            ``(results_dict, None)`` — the model object is not available
            across process boundaries so the second element is always
            ``None``.
        """
        if job_folder is None:
            raise RuntimeError(
                "job_folder is required for LLMFSDBenchmarkUtils: "
                "rank 0 writes results.json there for the parent to read.")

        # Determine number of GPUs from CUDA_VISIBLE_DEVICES
        # (already set by the benchmark framework for this worker).
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        nproc = len(cuda_visible.split(","))

        # Locate main.py relative to this file
        main_py = os.path.join(os.path.dirname(__file__), os.pardir, "main.py")
        main_py = os.path.abspath(main_py)

        accelerate_config = os.environ.get("BREVITAS_ACCELERATE_CONFIG")
        if accelerate_config is None:
            raise RuntimeError("BREVITAS_ACCELERATE_CONFIG is not set.")
        _validate_accelerate_config(accelerate_config)

        resolved_config_path = os.path.join(job_folder, "entrypoint_config.yaml")
        with open(resolved_config_path, "w") as config_file:
            yaml.safe_dump(_resolve_llm_config(args), config_file)
        argv = ["--config", resolved_config_path]
        if extra_args:
            argv += list(extra_args)
        # Forward job_folder so rank 0 writes results.json
        argv += ["--job-folder", job_folder]

        # Pick a free master port to avoid collisions when multiple
        # benchmark workers run concurrently on the same node.
        master_port = str(_find_free_port())

        cmd = [sys.executable, "-m", "accelerate.commands.launch"]
        cmd += ["--config_file", accelerate_config]
        if os.environ.get("BREVITAS_ACCELERATE_QUIET") == "1":
            cmd.append("--quiet")
        cmd += [
            "--num_processes",
            str(nproc),
            "--main_process_port",
            master_port,
            main_py,] + argv

        # The benchmark framework redirects sys.stdout / sys.stderr to
        # log files, but subprocess.run writes to OS-level file
        # descriptors.  Forward subprocess output to the current
        # sys.stdout / sys.stderr so that it lands in the benchmark's
        # stdout.out / stderr.out log files (and is available for
        # parse_log).
        proc = subprocess.run(cmd, env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"accelerate launch subprocess exited with return code {proc.returncode}")

        # Read results written by rank 0
        results_path = os.path.join(job_folder, "results.json")
        if not os.path.isfile(results_path):
            raise RuntimeError(
                f"results.json not found in {job_folder} — "
                "rank 0 may have crashed before writing results.")
        with open(results_path) as f:
            results = json.load(f)

        return results, None


class LLMFSDPGridBenchmark(BenchmarkUtils):
    entry_point_utils = LLMFSDPEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    # Pre-parse benchmark-specific args that are not known to the
    # shared benchmark framework (--accelerate-config, --quiet).
    _pre_parser = ArgumentParser(add_help=False)
    _pre_parser.add_argument(
        "--accelerate-config",
        type=str,
        required=True,
        dest="accelerate_config",
        help="Path to an Accelerate FSDP2 config YAML.")
    _pre_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Pass --quiet to accelerate launch to suppress its startup banner.")
    _pre_args, _remaining = _pre_parser.parse_known_args(sys.argv[1:])

    _validate_accelerate_config(_pre_args.accelerate_config)

    os.environ["BREVITAS_ACCELERATE_CONFIG"] = _pre_args.accelerate_config
    if _pre_args.quiet:
        os.environ["BREVITAS_ACCELERATE_QUIET"] = "1"

    LLMFSDPGridBenchmark.run(_remaining)
