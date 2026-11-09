# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import ArgumentTypeError
from argparse import Namespace
import importlib
import json
from numbers import Number
from pathlib import Path
import pprint
import sys
from typing import Any

from brevitas_examples.common.parse_utils import create_entrypoint_args_parser
from brevitas_examples.common.parse_utils import override_defaults
from brevitas_examples.common.parse_utils import parse_args

LM_EVAL_DEFAULT_TASKS = ['arc_challenge', 'arc_easy', 'winogrande', 'piqa']
LIGHTEVAL_DEFAULT_TASKS = ['arc:challenge|0', 'arc:easy|0', 'winogrande|0', 'piqa|0']
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
        help="Evaluation batch size or 'auto'. Default: %(default)s.")
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs used for tensor parallelism. Default: %(default)s.')
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='Fraction of GPU memory available to vLLM. Default: %(default)s.')
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
        help='Override the task generation length. Default: use the task setting.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory in which evaluation results are stored. Default: %(default)s.')
    parser.add_argument(
        '--seed', type=int, default=0, help='vLLM engine seed. Default: %(default)s.')
    return parser


def _validate_args(args: Namespace) -> Path:
    if not isinstance(args.model, str) or not args.model:
        raise ValueError('--model must be specified')
    if args.backend not in ('lm_eval', 'lighteval'):
        raise ValueError('--backend must be specified')
    if args.dtype not in ('auto', 'float16', 'bfloat16', 'float32'):
        raise ValueError(f'Unsupported --dtype: {args.dtype}')
    if not isinstance(args.tensor_parallel_size, int) or args.tensor_parallel_size < 1:
        raise ValueError('--tensor-parallel-size must be positive')
    if (args.batch_size != 'auto' and
        (not isinstance(args.batch_size, int) or args.batch_size < 1)):
        raise ValueError("--batch-size must be 'auto' or a positive integer")
    if (not isinstance(args.gpu_memory_utilization, (int, float)) or
            not 0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError('--gpu-memory-utilization must be in the interval (0, 1]')
    if (args.max_model_length is not None and
        (not isinstance(args.max_model_length, int) or args.max_model_length < 1)):
        raise ValueError('--max-model-length must be positive')
    if args.limit is not None and (not isinstance(args.limit, int) or args.limit < 1):
        raise ValueError('--limit must be positive')
    if args.thinking not in ('disabled', 'enabled'):
        raise ValueError('--thinking must be disabled or enabled')
    if args.max_new_tokens is not None and (not isinstance(args.max_new_tokens, int) or
                                            args.max_new_tokens < 1):
        raise ValueError('--max-new-tokens must be positive')
    if args.thinking == 'enabled':
        if not args.reasoning_start_tag or not args.reasoning_end_tag:
            raise ValueError('Reasoning tags must be non-empty when thinking is enabled')
        if args.reasoning_start_tag == args.reasoning_end_tag:
            raise ValueError('Reasoning start and end tags must differ')
    if not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError('--seed must be a non-negative integer')
    if args.tasks is not None and not isinstance(args.tasks, (str, list, tuple)):
        raise ValueError('--tasks must be a task name or a list of task names')

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f'Exported model directory does not exist: {model_path}')
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


def _register_brevitas_quantization() -> None:
    # Importing the manager registers quant_brevitas with vLLM.
    importlib.import_module('brevitas.export.inference.vLLM.manager')


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

    task_manager, is_generative = _lm_eval_task_mode(tasks)
    thinking_enabled = args.thinking == 'enabled'
    if thinking_enabled and not is_generative:
        raise ValueError('--thinking enabled is supported only for generative lm-eval tasks')

    model_args = {
        'pretrained': str(model_path),
        'quantization': 'quant_brevitas',
        'dtype': args.dtype,
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'seed': args.seed,
        'enable_thinking': thinking_enabled,}
    if thinking_enabled:
        model_args['think_end_token'] = args.reasoning_end_tag
    if args.max_model_length is not None:
        model_args['max_model_len'] = args.max_model_length
    if args.max_new_tokens is not None:
        model_args['max_gen_toks'] = args.max_new_tokens

    results = evaluator.simple_evaluate(
        model='vllm',
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
        apply_chat_template=is_generative,
        task_manager=task_manager,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    summary = _numeric_results(results)
    output_path = Path(args.output_dir) / 'lm_eval_results.json'
    with output_path.open('w', encoding='utf8') as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write('\n')
    pprint.pprint(summary)
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
        remove_reasoning_tags=args.thinking == 'enabled',
        reasoning_tags=[(args.reasoning_start_tag, args.reasoning_end_tag)])
    model_config_args = {
        'model_name': str(model_path),
        'dtype': args.dtype,
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
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


def evaluate(args: Namespace) -> dict[str, Any]:
    model_path = _validate_args(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    tasks = _tasks_for_backend(args)
    _register_brevitas_quantization()

    if args.backend == 'lm_eval':
        return run_lm_eval(args, model_path, tasks)
    return run_lighteval(args, model_path, tasks)


def main() -> None:
    parser = create_args_parser()
    overrides = override_defaults(sys.argv[1:])
    if overrides is None:
        overrides = {}
    elif not isinstance(overrides, dict):
        parser.error('--config must contain a YAML mapping')
    args, extra_args = parse_args(parser, sys.argv[1:], override_defaults=overrides)
    if extra_args:
        parser.error(f'unrecognized arguments: {" ".join(extra_args)}')
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
