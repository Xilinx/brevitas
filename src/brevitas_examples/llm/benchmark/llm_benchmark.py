# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
import ast
from functools import lru_cache
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import EntryPointUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.eval_vllm import parse_eval_args
from brevitas_examples.llm.llm_args import create_args_parser as create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args

# Arguments of the benchmark jobs that the LLM entrypoint does not know
VLLM_EVAL_KEYS = ('vllm_eval_config', 'vllm_eval_keep_export')
# Module of the harness of each backend of brevitas_vllm_eval
HARNESS_MODULES = {'lm_eval': 'lm_eval', 'lighteval': 'lighteval'}
# Run by the interpreter of an evaluation before any job starts, to check that it can evaluate
VLLM_EVAL_ENV_CHECK = """
import importlib
from importlib.metadata import entry_points
import sys

import vllm

importlib.import_module(sys.argv[1])
import brevitas_examples.llm.eval_vllm

# vLLM loads the quant_brevitas quantization method through this plugin
if not any(plugin.name == 'brevitas' for plugin in entry_points(group='vllm.general_plugins')):
    sys.exit('the vLLM plugin of Brevitas is not installed')
"""


def create_args_parser() -> ArgumentParser:
    """LLM entrypoint arguments, extended with the ones that only apply to benchmark jobs."""
    parser = create_llm_args_parser()
    parser.add_argument(
        '--vllm-eval-config',
        type=str,
        nargs='+',
        default=None,
        help=
        'YAML configuration of brevitas_vllm_eval (e.g., backend and tasks). If specified, the model '
        'exported with --export-target vllm is evaluated through vLLM after quantization, and the '
        'metrics are added to the results with a vllm_ prefix. With several configurations, the '
        'model is evaluated once with each of them, which is how to evaluate the likelihood and '
        'the generative tasks of the lm_eval backend, since it cannot mix them. Each evaluation runs '
        'with the python interpreter of its configuration, if specified. Default: %(default)s')
    parser.add_argument(
        '--vllm-eval-keep-export',
        action='store_true',
        help=
        'Keep the vLLM export in the job folder after the evaluation with --vllm-eval-config. By '
        'default, the export is deleted, even if the evaluation fails, to save disk space.')
    return parser


def vllm_eval_configs(args: Namespace) -> List[str]:
    """The evaluation configurations of a job. A benchmark YAML gives a plain string for one."""
    configs = getattr(args, 'vllm_eval_config', None)
    if configs is None:
        return []
    return [configs] if isinstance(configs, str) else list(configs)


def uses_vllm_eval(args: Namespace) -> bool:
    return len(vllm_eval_configs(args)) > 0


def vllm_eval_python(eval_args: Namespace) -> str:
    """The interpreter that runs an evaluation: the one of its configuration, or this one."""
    if eval_args.python is None:
        return sys.executable
    return os.path.expanduser(eval_args.python)


@lru_cache(maxsize=None)
def check_vllm_eval_env(python: str, backend: str) -> None:
    """Check that ``python`` can evaluate with ``backend``, once per interpreter and backend.

    It runs before any job starts, since a wrong environment would otherwise only fail after the
    quantization of every job.
    """
    harness = HARNESS_MODULES[backend]
    try:
        process = subprocess.run([python, '-c', VLLM_EVAL_ENV_CHECK, harness],
                                 capture_output=True,
                                 text=True)
    except OSError as exc:
        raise RuntimeError(
            f"The vLLM evaluation interpreter {python} cannot be run: {exc}") from exc
    if process.returncode != 0:
        lines = process.stderr.strip().splitlines()
        error = lines[-1] if lines else f"exit code {process.returncode}"
        raise RuntimeError(
            f"The environment of {python} cannot run the vLLM evaluation with the {backend} "
            f"backend ({error}). It needs vLLM, {harness} and Brevitas, which can be installed "
            f"from this repository with pip install -e \".[vllm_eval_{backend}]\".")


def validate_vllm_eval(args: Namespace) -> None:
    if uses_vllm_eval(args):
        assert args.export_target == 'vllm', "--vllm-eval-config requires --export-target vllm."
        # Fail before any job starts if an evaluation configuration or its environment is invalid
        eval_args = [parse_eval_args(['--config', config]) for config in vllm_eval_configs(args)]
        for config_args in eval_args:
            check_vllm_eval_env(vllm_eval_python(config_args), config_args.backend)


def vllm_export_dir(job_folder: str) -> str:
    # Each job exports into its own folder, so that parallel jobs do not overwrite each other
    return os.path.join(job_folder, 'vllm_export')


def evaluate_vllm_export(args: Namespace, job_folder: str) -> Dict[str, Any]:
    """Evaluate the vLLM export of a job, and return its metrics with a vllm_ prefix.

    The export is evaluated once with each configuration, by the python interpreter of the
    configuration, which can belong to another environment than the one of the quantization.
    vLLM runs in a separate process, which releases all its GPU memory when it terminates. Its
    output is redirected to the log of the job.
    """
    eval_dir = os.path.join(job_folder, 'vllm_eval')
    vllm_results = {}
    for config in vllm_eval_configs(args):
        python = vllm_eval_python(parse_eval_args(['--config', config]))
        sys.stdout.flush()
        sys.stderr.flush()
        subprocess.run([
            python,
            '-m',
            'brevitas_examples.llm.eval_vllm',
            '--config',
            config,
            '--model',
            vllm_export_dir(job_folder),
            '--output-dir',
            eval_dir],
                       stdout=sys.stdout,
                       stderr=sys.stderr,
                       check=True)
        # The next evaluation overwrites the results
        with open(os.path.join(eval_dir, 'results.json'), 'r') as f:
            vllm_results.update(json.load(f))
    return {f'vllm_{name}': value for name, value in vllm_results.items()}


def remove_vllm_export(args: Namespace, job_folder: str) -> None:
    export_dir = vllm_export_dir(job_folder)
    # The export does not exist if quantization failed
    if not args.vllm_eval_keep_export and os.path.isdir(export_dir):
        shutil.rmtree(export_dir)


class LLMEntryPointUtils(EntryPointUtils):

    argument_parser: ArgumentParser = create_args_parser()
    benchmark_args: Tuple[str, ...] = VLLM_EVAL_KEYS
    eval_metrics: List[str] = ["float_ppl", "quant_ppl", "quant_ear", "quant_kld"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        # Find the line containing Float PPL number
        float_ppl_line = re.search(r"Float perplexity \((.*?)\): (\d+\.\d+)", job_log)
        float_ppl = float(float_ppl_line.group(2)) if float_ppl_line is not None else None
        # Find the line containing Quant PPL number
        quant_ppl_line = re.search(r"Quantized perplexity \((.*?)\): (\d+\.\d+)", job_log)
        quant_ppl = float(quant_ppl_line.group(2)) if quant_ppl_line is not None else None
        quant_ear_line = re.search(
            r"Quantized expected acceptance rate @ top-\d+ \((.*?)\): (\d+\.\d+)", job_log)
        quant_ear = float(quant_ear_line.group(2)) if quant_ear_line is not None else None
        quant_kld_line = re.search(
            r"Quantized KL divergence @ top-\d+ \((.*?)\): (-?\d+\.\d+)", job_log)
        quant_kld = float(quant_kld_line.group(2)) if quant_kld_line is not None else None
        # Search for dictionaries in the log before the export. After it, the log has the output of
        # the vLLM evaluation, which runs in a subprocess and prints dictionaries that are not
        # Python literals
        export_line = re.search(r"^Export to .*$", job_log, re.MULTILINE)
        metrics_log = job_log if export_line is None else job_log[:export_line.start()]
        # Retrieve last dictionary, in case other dictionaries were printed to the log
        few_shot_eval = {}
        for dictionary in reversed(re.findall(r"({.*?})", metrics_log)):
            try:
                parsed = ast.literal_eval(dictionary)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                few_shot_eval = parsed
                break
        # Return the results from the log as a dictionary
        job_log_results = {
            "float_ppl": float_ppl,
            "quant_ppl": quant_ppl,
            "quant_ear": quant_ear,
            "quant_kld": quant_kld,
            **few_shot_eval,}
        return job_log_results

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_llm_args(args=args, extra_args=extra_args)
        validate_vllm_eval(args)

    @staticmethod
    def entrypoint_main(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        import torch

        from brevitas_examples.llm.main import quantize_llm

        if not uses_vllm_eval(args):
            return quantize_llm(args=args, extra_args=extra_args)

        args.export_prefix = vllm_export_dir(job_folder)
        try:
            results, model = quantize_llm(args=args, extra_args=extra_args)
            # Free the GPU memory held by the quantized model before vLLM allocates its own
            del model
            gc.collect()
            torch.cuda.empty_cache()
            vllm_results = evaluate_vllm_export(args, job_folder)
        finally:
            remove_vllm_export(args, job_folder)
        return {**results, **vllm_results}, None


class LLMGridBenchmark(BenchmarkUtils):
    entry_point_utils = LLMEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    LLMGridBenchmark.run(sys.argv[1:])
