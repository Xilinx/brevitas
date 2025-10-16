# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from contextlib import ExitStack
import logging
import os
import platform
import shutil
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

from datasets import Dataset
import onnx
from packaging import version
import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas import torch_version
from brevitas_examples.llm.llm_args import create_args_parser
from brevitas_examples.llm.main import fx_required
from brevitas_examples.llm.main import main as llm_main
from brevitas_examples.llm.main import quantize_llm
from tests.brevitas_examples.common import assert_layer_types
from tests.brevitas_examples.common import assert_layer_types_count
from tests.brevitas_examples.common import assert_metrics
from tests.brevitas_examples.common import get_default_args
from tests.brevitas_examples.common import parse_args_and_defaults
from tests.brevitas_examples.common import process_args_and_metrics
from tests.brevitas_examples.common import UpdatableNamespace
from tests.brevitas_examples.test_llm_cases import LLMPerplexityCases
from tests.brevitas_examples.test_llm_cases import LLMQuantLayerCountCases
from tests.brevitas_examples.test_llm_cases import LLMQuantLayerTypeCases
from tests.brevitas_examples.test_llm_cases import LLMRotationOptimizationCases
from tests.brevitas_examples.test_llm_cases import LLMRunCases
from tests.marker import jit_disabled_for_dynamic_quant_act
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

ATOL_PPL = 1e+01
RTOL_PPL = 1e-04

ATOL_ACC = 5e-1
RTOL_ACC = 1e-5


def mock_load_raw_dataset(dataset_name: str, split: str, seed: int = 42) -> Dataset:
    assert dataset_name == "c4", f"Expected dataset_name to be c4 but got {dataset_name} instead"
    assert split in ["train", "validation"], f"Expected split to be 'train' or 'validation' but got "
    # Contains information from allenai/c4 (https://huggingface.co/datasets/allenai/c4) which is made available under the ODC Attribution License.
    C4_TEXTS = [
        "Luxembourg's professional networking group for women will host a discussion about promoting Luxembourg abroad.\n(JB) Luxembourg's female only professional networking group will host a discussion about promoting Luxembourg abroad.\nSpeaker Carole Tompers, who is responsible for promoting the Made in Luxembourg products and services to foreign markets, will take guests on a whistle stop tour of the country's key assets.\nHer speech will explore the Nations Brand Index 2010, delve into what makes a Luxembourg brand and suggest ways of strengthening and promoting existing brands abroad.\nMs Tompers has a strong track record in marketing and communications. She currently serves as Secretary General at Luxembourg for Business.\nShe has previously worked on promotional projects with various ministries, the Chamber of Commerce, the Office Ducroire, the National Credit and Investment Corporation, the Chamber of Crafts and Luxembourg's Business Federation.\nThe event is organised by the Network at the Sofitel Kirchberg on November 16, from 7.30pm."
    ]
    return Dataset.from_dict({
        "text": C4_TEXTS,})


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


# Check that all args in args are used
def validate_args(parser: ArgumentParser, args: Namespace) -> None:
    a, da = parse_args_and_defaults(args, parser)
    for k in a.keys():
        assert k in da.keys(), f"Key {k} does not seem to be a valid argument for `quantize_llm`"
    if args.replace_rmsnorm:
        if torch_version < version.parse('2.4'):
            pytest.skip("Replacing RMSNorm requires torch 2.4+ or greater")
    if args.gpxq_block_name == "model.layers" and args.learned_round is not None and "opt" in args.model.lower(
    ):
        pytest.skip(
            f"OPT-style model {args.model} not support with learned_round={args.learned_round} with block module named {args.gpxq_block_name}"
        )

    use_fx = fx_required(args) or args.rotation == 'fused_no_fx'
    #if use_fx and not model_with_ppl.supports_fx:
    #    pytest.xfail(f"{model_with_ppl.name} does not support FX")
    if args.input_scale_type == 'dynamic' and config.JIT_ENABLED:
        pytest.skip("Dynamic activation not compatible with JIT")
    if platform.system() == 'Windows' and use_fx:
        pytest.skip("Skipping dynamo + Windows")

    if args.weight_param_method == 'hqo' and config.JIT_ENABLED:
        pytest.skip("Local loss mode requires JIT to be disabled")


@pytest.fixture
def parser() -> ArgumentParser:
    return create_args_parser()


@pytest.fixture
def main(parser) -> Callable:

    def wrapper_main(
            args: UpdatableNamespace,
            extra_args: Optional[List[str]] = None) -> Tuple[torch.nn.Module, Dict[str, float]]:
        with patch('brevitas_examples.llm.llm_quant.data_utils.load_raw_dataset',
                   mock_load_raw_dataset):
            # Validate the arguments before running the entrypoint
            validate_args(parser, args)
            results, model = quantize_llm(args, extra_args=extra_args)
        # Return the results along with the model
        return results, model

    return wrapper_main


@pytest_cases.fixture()
def default_run_args(parser: ArgumentParser, request):
    args = get_default_args(parser)
    args.nsamples = 2
    args.seqlen = 2
    args.model = "hf-internal-testing/tiny-random-MistralForCausalLM"
    args.dataset = "c4"
    args.eval = True
    #args.checkpoint = ptid2pathname(request.node.nodeid) + ".pth" # Example filename which won't clash
    args.export_prefix = ptid2pathname(request.node.nodeid)
    args.weight_bit_width = 8
    args.weight_quant_granularity = "per_channel"  # "per_tensor", "per_channel", "per_group".
    args.input_bit_width = 8
    args.act_calibration = True
    args.dtype = "float32"
    return args


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_and_metrics", cases=LLMRunCases)
def test_small_models_run_args(caplog, args_and_metrics, main):
    caplog.set_level(logging.INFO)
    args, extra_args, _ = args_and_metrics
    main(args, extra_args)


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_and_metrics", cases=LLMPerplexityCases)
def test_small_models_ppl(caplog, args_and_metrics, main):
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_and_metrics
    results, _ = main(args, extra_args)
    assert_metrics(results, exp_metrics, atol=ATOL_PPL, rtol=RTOL_PPL)


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_and_layer_types", cases=LLMQuantLayerTypeCases)
def test_small_models_quant_layer(caplog, args_and_layer_types, main):
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_and_layer_types
    _, model = main(args, extra_args)
    assert_layer_types(model, exp_metrics["exp_layer_types"])


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_and_layer_types_count", cases=LLMQuantLayerCountCases)
def test_small_models_quant_layer_types_count(caplog, args_and_layer_types_count, main):
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_and_layer_types_count
    _, model = main(args, extra_args)
    assert_layer_types_count(model, exp_metrics["exp_layer_types_count"])


@pytest_cases.fixture(
    ids=[
        "mistral-kv-quant-fx-sdpa",
        "mistral-kv-quant-functional-sdpa",
        "mistral-kv-quant-eager-sdpa"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "input_quant_granularity": "per_row",
            "attn_quant_granularity": "per_group",
            "input_group_size": 32,
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "quant_sdpa": "fx",
            "attn_quant_config": "kv",
            "attn_quant_type": "asym"},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "input_quant_granularity": "per_row",
            "attn_quant_granularity": "per_group",
            "input_group_size": 32,
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "quant_sdpa": "functional",
            "attn_quant_config": "kv",
            "attn_quant_type": "asym"},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "input_quant_granularity": "per_row",
            "attn_quant_granularity": "per_group",
            "input_group_size": 32,
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "quant_sdpa": "eager",
            "attn_quant_config": "kv",
            "attn_quant_type": "asym"},])
def layer_args_hyperparam(default_run_args, request):
    yield process_args_and_metrics(default_run_args, request.param)


@pytest.mark.llm
@jit_disabled_for_dynamic_quant_act()
def test_small_models_quant_layer_hyperparam(caplog, layer_args_hyperparam, main):
    from brevitas.nn import QuantScaledDotProductAttention as QuantSDPA
    from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
    caplog.set_level(logging.INFO)
    args, _, _ = layer_args_hyperparam

    use_fx = fx_required(args) or args.rotation == 'fused_no_fx'

    if platform.system() == 'Windows' and use_fx:
        pytest.skip("Skipping dynamo + Windows")

    _, model = main(args)
    quant_sdpa = []
    for m in model.modules():
        if isinstance(m, QuantSDPA):
            quant_sdpa.append(m)

    first_sdpa = quant_sdpa[0]

    # Check that Q/Softmax quantization is disabled
    assert first_sdpa.q_scaled_quant.act_quant.fused_activation_quant_proxy is None
    assert first_sdpa.attn_output_weights_quant.act_quant.fused_activation_quant_proxy is None
    # NOTE: We assume that asym == unsigned. This might change in the future.
    assert not first_sdpa.v_quant.act_quant.is_signed
    assert not first_sdpa.k_transposed_quant.act_quant.is_signed
    # Check for groupwise activation quantization
    assert isinstance(first_sdpa.v_quant.act_quant, GroupwiseActQuantProxyFromInjector)
    assert isinstance(first_sdpa.k_transposed_quant.act_quant, GroupwiseActQuantProxyFromInjector)
    assert first_sdpa.v_quant.act_quant.group_size == args.input_group_size
    assert first_sdpa.k_transposed_quant.act_quant.group_size == args.input_group_size
    # Functional quantization uses one shared quant block for everything
    if args.quant_sdpa == "fx" or args.quant_sdpa == "eager":
        assert len(quant_sdpa) == 2
    elif args.quant_sdpa == "functional":
        assert len(quant_sdpa) == 1


@pytest_cases.fixture(
    ids=[
        "qcdq-asym",
        "qcdq-sym",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "quantize_weight_zero_point": True,
            "quantize_input_zero_point": True,
            "export_target": "onnx_qcdq",},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "weight_quant_type": "sym",
            "input_quant_type": "sym",
            "export_target": "onnx_qcdq",},])
def onnx_export_args(default_run_args, request):
    yield process_args_and_metrics(default_run_args, request.param)


@pytest.mark.llm
@jit_disabled_for_export()
@requires_pt_ge('2.5')
def test_small_models_onnx_export(caplog, onnx_export_args, main):
    caplog.set_level(logging.INFO)
    args, _, _ = onnx_export_args
    main(args)
    onnx.load(os.path.join(args.export_prefix, "model.onnx"))
    shutil.rmtree(args.export_prefix)


@pytest_cases.fixture(
    ids=["auto", "float16", "bfloat16"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "dtype": "auto"},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "dtype": "float16"},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "dtype": "float16"},])
def dtype_args(default_run_args, request):
    yield process_args_and_metrics(default_run_args, request.param)


@pytest.mark.llm
def test_small_models_dtype(caplog, dtype_args, main):
    caplog.set_level(logging.INFO)
    args, _, _ = dtype_args
    _, model = main(args)
    # "auto" dtype for "hf-internal-testing/tiny-random-LlamaForCausalLM" is float32
    expected_dtype = torch.float32 if args.dtype == "auto" else getattr(torch, args.dtype)
    dtype = next(model.parameters()).dtype
    assert expected_dtype == dtype, f"Expected dtype of the model parameters to be {expected_dtype} but got {dtype}."


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_layer_count_and_ppl", cases=LLMRotationOptimizationCases)
def test_small_models_rotation_optimization_ppl(caplog, args_layer_count_and_ppl, main):
    if platform.system() != "Linux":
        pytest.skip("Skipping dynamo + windows/macos")
    # Tolerances are stricter for this test, to ensure that it does not pass
    # with non-optimized quantized perplexities
    RTOL_ROT, ATOL_ROT = 1e-05, 2.
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_layer_count_and_ppl
    # Drop the unnecesary entries from exp_metrics
    del exp_metrics["exp_layer_types_count"]
    results, _ = main(args, extra_args)
    assert_metrics(results, exp_metrics, atol=ATOL_ROT, rtol=RTOL_ROT)


@pytest.mark.llm
@pytest_cases.parametrize_with_cases("args_layer_count_and_ppl", cases=LLMRotationOptimizationCases)
def test_small_models_rotation_optimization_layer_count(caplog, args_layer_count_and_ppl, main):
    if platform.system() != "Linux":
        pytest.skip("Skipping dynamo + windows/macos")
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_layer_count_and_ppl
    with patch('brevitas_examples.llm.main.fuse_parametrizations', lambda model: model):
        _, model = main(args, extra_args)
    assert_layer_types_count(model, exp_metrics["exp_layer_types_count"])


@pytest.mark.llm
@pytest_cases.parametrize(
    "kwargs",
    [
        {
            "yaml_file_path":
                "./tests/brevitas_examples/llm_test_template.yml",
            "expected_extra_args": [
                "--learning_rate",
                "1.5",
                "--lr_scheduler_type",
                "cosine",
                "--save_safetensors",
                "False"],},],
    ids=lambda kwargs: kwargs["yaml_file_path"])
def test_parse_yaml_trainer_arguments(caplog, kwargs):
    caplog.set_level(logging.INFO)
    yaml_file_path = kwargs["yaml_file_path"]
    expected_extra_args = kwargs["expected_extra_args"]
    extra_args_keys = [expected_extra_args[i][2:] for i in range(0, len(expected_extra_args), 2)]

    def quantize_llm_assert_args(args, extra_args=None):
        for key in extra_args_keys:
            assert key not in args, f"Key {key} should not be known by the parser"
        assert extra_args == expected_extra_args, f"Expected extra arguments {expected_extra_args} but got {extra_args}"

    # Run the argument parsing logic of the LLM entrypoint
    with patch("brevitas_examples.llm.main.quantize_llm", quantize_llm_assert_args):
        with patch("brevitas_examples.llm.main.sys.argv", ["main.py", "--config", yaml_file_path]):
            llm_main()


@pytest_cases.fixture(
    ids=["lighteval"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "few_shot_eval": "lighteval",
            "few_shot_override_batch_size": 16,
            "few_shot_tasks": [
                "leaderboard|arc:challenge|0|0",
                "leaderboard|winogrande|0|0",
                "lighteval|arc:easy|0|0",
                "leaderboard|hellaswag|0|0",],
            "few_shot_zeroshot": True,
            "imports": ["lighteval"],
            "all_acc": 0.375,},])
def few_shot_eval_args(default_run_args, request):
    # Skip cases for which the LM evaluation library has not been installed
    for lib in request.param["imports"]:
        pytest.importorskip(lib, reason=f"`{lib}` needs to be installed.")
    del request.param["imports"]

    yield process_args_and_metrics(
        default_run_args, request.param, extra_keys=["imports", "all_acc"])


@pytest.mark.llm
def test_few_shot_eval(caplog, few_shot_eval_args, main):
    caplog.set_level(logging.INFO)
    args, _, exp_metrics = few_shot_eval_args

    with ExitStack() as ctx_stack:
        # Patch LM eval calls when needed
        if args.few_shot_eval == "lighteval":
            from brevitas_examples.llm.eval_lighteval import run_lighteval
            max_samples = args.few_shot_override_batch_size

            def mock_run_lighteval(*args, **kwargs):
                kwargs["max_samples"] = max_samples
                return run_lighteval(*args, **kwargs)

            # Patch the call to `run_lighteval`
            ctx_stack.enter_context(
                patch(
                    'brevitas_examples.llm.eval_lighteval.run_lighteval',
                    side_effect=mock_run_lighteval))

        results, _ = main(args)

    # Verify that LM eval metrics match. `strict` is set to False, as
    # only a subset of metrics are checked.
    assert_metrics(results, exp_metrics, atol=ATOL_ACC, rtol=RTOL_ACC, strict=False)
