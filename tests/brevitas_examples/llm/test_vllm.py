# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
from brevitas_examples.llm.eval_vllm import _parallel_sizes
from brevitas_examples.llm.eval_vllm import parse_eval_args
from brevitas_examples.llm.llm_args import create_args_parser
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.main import quantize_and_evaluate_llm

pytestmark = pytest.mark.llm

INT_PER_ROW_INPUT = [
    '--input-bit-width',
    '8',
    '--input-scale-type',
    'dynamic',
    '--input-quant-granularity',
    'per_row']
FP8_PER_ROW_INPUT = INT_PER_ROW_INPUT + ['--input-quant-format', 'float_ocp_e4m3']
KV_QUANT = ['--quant-sdpa', 'eager', '--attn-quant-config', 'kv']


def write_eval_config(tmp_path, **config):
    path = tmp_path / 'eval.yaml'
    path.write_text(yaml.dump(config))
    return str(path)


def parse_llm_args(*args):
    return create_args_parser().parse_args(list(args))


@pytest.mark.parametrize(
    'extra_args, error',
    [
        (['--quantize-first-last-layer'], None),
        (['--rotation', 'layerwise', '--rotation-block-size', '32'], None),
        (KV_QUANT + ['--input-bit-width', '8', '--act-calibration'], None),
        (FP8_PER_ROW_INPUT, None),
        (['--rotation-layers-to-expand', 'down_proj'], 'expanded rotations'),
        (['--quant-sdpa', 'fx'], '--quant-sdpa eager'),
        (['--quant-sdpa', 'eager'], 'softmax output'),
        (KV_QUANT + FP8_PER_ROW_INPUT, 'static per-tensor'),
        (INT_PER_ROW_INPUT, 'float formats'),],
    ids=[
        'first_last_layer',
        'block_online_rotation',
        'kv_quant',
        'fp8_per_row_input',
        'expanded_rotation',
        'fx_sdpa',
        'softmax_quant',
        'dynamic_attn',
        'int_per_row_input'])
def test_vllm_export_validation(extra_args, error):
    args = parse_llm_args('--export-target', 'vllm', *extra_args)
    if error is None:
        validate(args)
    else:
        with pytest.raises(AssertionError, match=error):
            validate(args)


def test_vllm_eval_config_validation(tmp_path):
    args = parse_llm_args(
        '--export-target',
        'vllm',
        '--vllm-eval-config',
        write_eval_config(tmp_path, backend='lm_eval'))
    validate(args)

    args.export_target = None
    with pytest.raises(AssertionError, match='--export-target vllm'):
        validate(args)

    # An invalid evaluation configuration must fail before quantization starts
    args.export_target = 'vllm'
    args.vllm_eval_config = write_eval_config(tmp_path, tasks=['arc_easy'])
    with pytest.raises(ValueError, match='--backend must be specified'):
        validate(args)


@pytest.mark.parametrize(
    'config, error',
    [
        ({
            'tasks': ['arc_easy']}, '--backend must be specified'),
        ({
            'backend': 'lmeval'}, 'Invalid backend'),
        ({
            'backend': 'lm_eval', 'parallelism': 'dta'}, 'Invalid parallelism'),
        ({
            'backend': 'lm_eval', 'unknown_option': 1}, 'unrecognized arguments'),
        ({
            'backend': 'lighteval', 'cpu_offload_gb': 16}, 'only by the lm_eval backend'),],
    ids=[
        'missing_backend', 'mistyped_backend', 'mistyped_parallelism', 'unknown_option', 'offload'])
def test_parse_eval_args_invalid(tmp_path, config, error):
    with pytest.raises(ValueError, match=error):
        parse_eval_args(['--config', write_eval_config(tmp_path, **config)])


@pytest.mark.parametrize(
    'parallelism, data_parallel_size, pipeline_parallel_size', [(None, 1, 1), ('data', 8, 1),
                                                                ('pipeline', 1, 8)])
def test_parallelism_uses_visible_gpus(
        tmp_path, parallelism, data_parallel_size, pipeline_parallel_size):
    config = write_eval_config(tmp_path, backend='lm_eval', parallelism=parallelism)
    with patch('torch.cuda.device_count', return_value=8):
        sizes = _parallel_sizes(parse_eval_args(['--config', config]))
    assert sizes == {
        'data_parallel_size': data_parallel_size, 'pipeline_parallel_size': pipeline_parallel_size}


@pytest.fixture
def vllm_eval_run():
    """Replaces the quantization and the evaluation, which runs in a subprocess."""

    def quantize_llm(args, extra_args=None):
        # Emulate the export
        with open(os.path.join(args.export_prefix, 'brevitas_config.json'), 'w') as f:
            f.write('{}')
        return {'quant_ppl': 10.0}, object()

    def run(cmd, **kwargs):
        # The export exists while vLLM runs, and its metrics are stored in the output folder
        assert os.path.isfile(os.path.join(cmd[cmd.index('--model') + 1], 'brevitas_config.json'))
        output_dir = cmd[cmd.index('--output-dir') + 1]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump({'arc_easy_acc,none': 0.5}, f)

    with patch('brevitas_examples.llm.main.quantize_llm', quantize_llm), \
         patch('brevitas_examples.llm.main.subprocess.run', side_effect=run) as run_mock:
        yield run_mock


@pytest.mark.parametrize('keep_export', [False, True], ids=['delete_export', 'keep_export'])
@pytest.mark.parametrize(
    'entrypoint',
    [
        lambda args,
        root: quantize_and_evaluate_llm(args, export_root=root),
        lambda args,
        root: LLMEntryPointUtils.entrypoint_main(args, job_folder=root)],
    ids=['cli', 'benchmark'])
def test_vllm_eval_after_quantization(tmp_path, vllm_eval_run, entrypoint, keep_export):
    args = parse_llm_args(
        '--export-target',
        'vllm',
        '--vllm-eval-config',
        write_eval_config(tmp_path, backend='lm_eval'),
        *(['--vllm-eval-keep-export'] if keep_export else []))
    root = str(tmp_path / 'job')

    results, model = entrypoint(args, root)

    cmd = vllm_eval_run.call_args.args[0]
    export_dir = cmd[cmd.index('--model') + 1]
    assert cmd[:3] == [sys.executable, '-m', 'brevitas_examples.llm.eval_vllm']
    assert cmd[cmd.index('--config') + 1] == args.vllm_eval_config
    assert os.path.dirname(export_dir) == root
    assert cmd[cmd.index('--output-dir') + 1] == os.path.join(root, 'vllm_eval')
    assert model is None
    assert results['quant_ppl'] == 10.0
    assert results['vllm_arc_easy_acc,none'] == 0.5
    assert os.path.isdir(export_dir) == keep_export
    assert results.get('vllm_export_dir') == (export_dir if keep_export else None)


def test_vllm_eval_deletes_export_on_failure(tmp_path, vllm_eval_run):
    vllm_eval_run.side_effect = subprocess.CalledProcessError(1, 'brevitas_vllm_eval')
    args = parse_llm_args(
        '--export-target',
        'vllm',
        '--vllm-eval-config',
        write_eval_config(tmp_path, backend='lm_eval'))
    root = str(tmp_path / 'job')

    with pytest.raises(subprocess.CalledProcessError):
        quantize_and_evaluate_llm(args, export_root=root)

    assert os.listdir(root) == []
