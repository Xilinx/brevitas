# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import ArgumentTypeError
from argparse import Namespace
import json
from numbers import Number
import os
from pathlib import Path
import pprint
import sys
from typing import Any

from brevitas_examples.common.parse_utils import create_entrypoint_args_parser
from brevitas_examples.common.parse_utils import override_defaults
from brevitas_examples.common.parse_utils import parse_args

LM_EVAL_DEFAULT_TASKS = ['arc_challenge', 'arc_easy', 'winogrande', 'piqa']
# piqa_lm_eval is the log-likelihood PIQA task defined in eval_lighteval.py
LIGHTEVAL_DEFAULT_TASKS = ['arc:challenge|0', 'arc:easy|0', 'winogrande|0', 'piqa_lm_eval|0']
LIGHTEVAL_TASK_ALIASES = {
    'arc_challenge': 'arc:challenge',
    'arc_easy': 'arc:easy',}


def _batch_size(value: str) -> str | int:
    if value == 'auto':
        return value
    try:
        value = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("batch size must be 'auto' or a positive integer") from exc
    if value < 1:
        raise ArgumentTypeError("batch size must be 'auto' or a positive integer")
    return value


def create_args_parser() -> ArgumentParser:
    parser = create_entrypoint_args_parser(
        description='Zero-shot evaluation of a Brevitas-exported vLLM model')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to a model exported with --export-target vllm.')
    parser.add_argument(
        '--backend',
        type=str,
        default=None,
        choices=['lm_eval', 'lighteval'],
        help='Evaluation backend. This argument must be specified.')
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Tasks to evaluate. Backend-specific zero-shot defaults are used when omitted.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Data type used by vLLM. Default: %(default)s.')
    parser.add_argument(
        '--batch-size',
        type=_batch_size,
        default='auto',
        help=
        "Maximum number of sequences processed concurrently by vLLM (max_num_seqs), or 'auto' to "
        "use the backend default. Default: %(default)s.")
    parser.add_argument(
        '--add-bos-token',
        action='store_true',
        help='Prepend the BOS token to lm_eval requests, as done by brevitas_ptq_llm with '
        '--bos-preprocessing. LightEval always adds special tokens.')
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='Fraction of GPU memory available to vLLM. Default: %(default)s.')
    parser.add_argument(
        '--parallelism',
        type=str,
        default=None,
        choices=[None, 'data', 'pipeline'],
        help=
        'How to use the visible GPUs (e.g., CUDA_VISIBLE_DEVICES). data runs a replica of the model '
        'on each GPU, each evaluating a share of the requests: it is the fastest option if the model '
        'fits on a single GPU. pipeline splits the model layers across the GPUs, for models that do '
        'not fit on a single GPU. By default, a single GPU is used. Default: %(default)s.')
    parser.add_argument(
        '--cpu-offload-gb',
        type=float,
        default=0,
        help=
        'GiB of weights per GPU offloaded to CPU memory and streamed to the GPU at every forward '
        'pass. Only supported by the lm_eval backend. Default: %(default)s.')
    parser.add_argument(
        '--max-model-length',
        type=int,
        default=None,
        help='Maximum model sequence length. By default vLLM infers it from the model.')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of samples per task. Intended for smoke tests.')
    parser.add_argument(
        '--thinking',
        type=str,
        choices=['disabled', 'enabled'],
        default='disabled',
        help='Thinking mode for generative tasks. Default: %(default)s.')
    parser.add_argument(
        '--reasoning-start-tag',
        type=str,
        default='<think>',
        help='Opening tag for reasoning output. Default: %(default)s.')
    parser.add_argument(
        '--reasoning-end-tag',
        type=str,
        default='</think>',
        help='Closing tag for reasoning output. Default: %(default)s.')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=None,
        help='Override the generation length of every task. Default: use the task setting.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory in which evaluation results are stored. Default: %(default)s.')
    parser.add_argument(
        '--seed', type=int, default=0, help='vLLM engine seed. Default: %(default)s.')
    parser.add_argument(
        '--python',
        type=str,
        default=None,
        help='Python interpreter of the environment in which the benchmarks run this evaluation '
        '(e.g., /path/to/env/bin/python), where vLLM, the evaluation harness and Brevitas are '
        'installed. It can differ from the environment used for quantization. brevitas_vllm_eval '
        'itself ignores it, and runs in the environment it is started from. Default: the '
        'interpreter of the benchmark.')
    return parser


def parse_eval_args(argv: list[str]) -> Namespace:
    """Parse and check the evaluation arguments, which can also be provided through --config."""
    parser = create_args_parser()
    overrides = override_defaults(argv)
    if overrides is None:
        overrides = {}
    elif not isinstance(overrides, dict):
        raise ValueError('--config must contain a YAML mapping')
    args, extra_args = parse_args(parser, argv, override_defaults=overrides)
    if extra_args:
        raise ValueError(f'unrecognized arguments: {" ".join(extra_args)}')
    # The backend defaults to None so that it can be provided through --config
    if args.backend is None:
        raise ValueError('--backend must be specified')
    # Values provided through --config are not checked by the parser, and a mistyped choice
    # (e.g., the backend) would otherwise silently select the wrong behavior
    for action in parser._actions:
        if action.choices is not None and getattr(args, action.dest) not in action.choices:
            raise ValueError(
                f'Invalid {action.dest}: {getattr(args, action.dest)}. '
                f'Choose from {list(action.choices)}')
    if args.backend == 'lighteval' and args.cpu_offload_gb > 0:
        raise ValueError(
            '--cpu-offload-gb is supported only by the lm_eval backend, LightEval does not expose '
            'vLLM CPU offloading. Use --parallelism pipeline instead.')
    return args


def _parallel_sizes(args: Namespace) -> dict[str, int]:
    """Data or pipeline parallel size, as expected by both backends, using all the visible GPUs."""
    import torch

    num_gpus = torch.cuda.device_count() if args.parallelism is not None else 1
    return {
        'data_parallel_size': num_gpus if args.parallelism == 'data' else 1,
        'pipeline_parallel_size': num_gpus if args.parallelism == 'pipeline' else 1}


def _validate_model(args: Namespace) -> Path:
    # The model defaults to None so that it can be provided through --config
    if args.model is None:
        raise ValueError('--model must be specified')

    model_path = Path(args.model).expanduser()
    config_path = model_path / 'brevitas_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'{config_path} was not found. The model must be exported with --export-target vllm.')
    return model_path.resolve()


def _lighteval_zero_shot_task(task: str) -> str:
    if not isinstance(task, str) or not task:
        raise ValueError('LightEval task names must be non-empty strings')
    if '|' not in task:
        return f"{LIGHTEVAL_TASK_ALIASES.get(task, task)}|0"

    parts = task.split('|')
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid LightEval task '{task}'; expected task|0 or suite|task|0")
    try:
        num_fewshot = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot determine the few-shot count in LightEval task '{task}'") from exc
    if num_fewshot != 0:
        raise ValueError(f"LightEval task '{task}' is not zero-shot")
    return task


def _tasks_for_backend(args: Namespace) -> list[str]:
    configured_tasks = [args.tasks] if isinstance(args.tasks, str) else args.tasks
    if args.backend == 'lm_eval':
        return list(configured_tasks or LM_EVAL_DEFAULT_TASKS)
    tasks = configured_tasks or LIGHTEVAL_DEFAULT_TASKS
    return [_lighteval_zero_shot_task(task) for task in tasks]


def _numeric_results(results: dict[str, Any]) -> dict[str, int | float]:
    summary = {}
    for task_name, metrics in results.get('results', {}).items():
        for metric_name, value in metrics.items():
            if isinstance(value, Number):
                summary[f'{task_name}_{metric_name}'] = (
                    value.item() if hasattr(value, 'item') else value)
    return summary


def _lm_eval_task_mode(tasks: list[str]):
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    loaded_tasks = task_manager.load(tasks)['tasks']
    task_types = {
        task_name: task.get_config('output_type') for task_name, task in loaded_tasks.items()}
    unsupported = {
        name: output_type for name,
        output_type in task_types.items() if output_type not in (
            'generate_until', 'loglikelihood', 'multiple_choice', 'loglikelihood_rolling')}
    if unsupported:
        raise ValueError(f'Unsupported lm-eval task output types: {unsupported}')

    has_generation = any(value == 'generate_until' for value in task_types.values())
    has_likelihood = any(value != 'generate_until' for value in task_types.values())
    if has_generation and has_likelihood:
        raise ValueError(
            'lm_eval cannot apply task-specific chat formatting to a mixed generative and '
            'likelihood task set. Run the task types separately or use --backend lighteval.')
    return task_manager, has_generation


def run_lm_eval(args: Namespace, model_path: Path, tasks: list[str]) -> dict[str, Any]:
    from lm_eval import evaluator
    from transformers import AutoTokenizer

    task_manager, is_generative = _lm_eval_task_mode(tasks)
    # As in LightEval, generative tasks use the chat template only if the model defines one
    use_chat_template = is_generative and AutoTokenizer.from_pretrained(
        model_path).chat_template is not None
    thinking_enabled = args.thinking == 'enabled'
    if thinking_enabled and not use_chat_template:
        raise ValueError(
            '--thinking enabled requires generative lm-eval tasks and a model chat template')

    model_args = {
        'pretrained': str(model_path),
        'quantization': 'quant_brevitas',
        'dtype': args.dtype,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        **_parallel_sizes(args),
        'cpu_offload_gb': args.cpu_offload_gb,
        'seed': args.seed,
        'add_bos_token': args.add_bos_token,
        'enable_thinking': thinking_enabled,
        # Brevitas quantizers run as regular PyTorch code, so we skip compilation and CUDA graphs,
        # as LightEval does.
        'enforce_eager': True,}
    if thinking_enabled:
        model_args['think_end_token'] = args.reasoning_end_tag
    if args.max_model_length is not None:
        model_args['max_model_len'] = args.max_model_length
    if args.batch_size != 'auto':
        model_args['max_num_seqs'] = args.batch_size
    # Unlike max_gen_toks in model_args, gen_kwargs overrides the generation length set by tasks
    gen_kwargs = {'max_gen_toks': args.max_new_tokens} if args.max_new_tokens is not None else None

    results = evaluator.simple_evaluate(
        model='vllm',
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        # All requests are submitted at once and vLLM schedules them, up to max_num_seqs
        batch_size='auto',
        limit=args.limit,
        log_samples=False,
        apply_chat_template=use_chat_template,
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
    )
    return results


def run_lighteval(args: Namespace, model_path: Path, tasks: list[str]) -> dict[str, Any]:
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.pipeline import ParallelismManager
    from lighteval.pipeline import Pipeline
    from lighteval.pipeline import PipelineParameters

    from brevitas_examples.llm.lighteval_prompt import BrevitasPromptManager

    evaluation_tracker = EvaluationTracker(output_dir=args.output_dir, save_details=True)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        max_samples=args.limit,
        # Same custom tasks (e.g., piqa_lm_eval) as brevitas_ptq_llm
        custom_tasks_directory=str(Path(__file__).with_name('eval_lighteval.py')),
        remove_reasoning_tags=args.thinking == 'enabled',
        reasoning_tags=[(args.reasoning_start_tag, args.reasoning_end_tag)])
    model_config_args = {
        'model_name': str(model_path),
        'dtype': args.dtype,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        **_parallel_sizes(args),
        'max_model_length': args.max_model_length,
        'quantization': 'quant_brevitas',
        'seed': args.seed,}
    if args.batch_size != 'auto':
        model_config_args['max_num_seqs'] = args.batch_size
    model_config = VLLMModelConfig(**model_config_args)
    if args.max_new_tokens is not None:
        model_config.generation_parameters.max_new_tokens = args.max_new_tokens
    pipeline = Pipeline(
        tasks=','.join(tasks),
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    if args.thinking == 'enabled' and not pipeline.model.use_chat_template:
        raise ValueError(
            'Thinking requires a tokenizer chat template, but this model does not define one')
    pipeline.model.prompt_manager = BrevitasPromptManager(
        use_chat_template=pipeline.model.use_chat_template,
        tokenizer=pipeline.model.tokenizer,
        system_prompt=model_config.system_prompt,
        generation_thinking=args.thinking == 'enabled')
    # LightEval's cache key does not include our task-aware prompt or thinking policy.
    pipeline.model._cache = None
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()
    return pipeline.get_results()


def evaluate(args: Namespace) -> dict[str, int | float]:
    """Evaluate the model and store the numeric metrics of both backends in results.json."""
    # vLLM forks its engine process unless CUDA is initialized, but a query like
    # torch.cuda.is_available() in this process (e.g., by the libraries of a task) already prevents
    # a forked process from using CUDA. A spawned one starts clean.
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    model_path = _validate_model(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    tasks = _tasks_for_backend(args)

    if args.backend == 'lm_eval':
        results = run_lm_eval(args, model_path, tasks)
    else:
        results = run_lighteval(args, model_path, tasks)
    summary = _numeric_results(results)
    with (output_dir / 'results.json').open('w', encoding='utf8') as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write('\n')
    pprint.pprint(summary)
    return summary


def main() -> None:
    try:
        evaluate(parse_eval_args(sys.argv[1:]))
    except (FileNotFoundError, ValueError) as exc:
        create_args_parser().error(str(exc))


if __name__ == '__main__':
    main()
