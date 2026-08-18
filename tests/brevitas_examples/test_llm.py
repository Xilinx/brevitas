# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
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
from torch import nn

from brevitas import config
from brevitas import torch_version
from brevitas.graph.equalize import _compute_rotations
from brevitas.graph.equalize import Region
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.llm_args import create_args_parser
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.llm_quant.ln_affine_merge import rmsnorm_patch
from brevitas_examples.llm.llm_quant.parse_utils import parse_custom_trainer
from brevitas_examples.llm.llm_quant.rotation_optimization import parse_rotation_optimization_args
from brevitas_examples.llm.llm_quant.trainer_utils import _build_optimizers_from_configs
from brevitas_examples.llm.llm_quant.trainer_utils import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY
from brevitas_examples.llm.main import _functional_quant_map
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
from tests.conftest import SEED
from tests.marker import jit_disabled_for_dynamic_quant_act
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

ATOL_PPL = 1e+01
RTOL_PPL = 1e-04

ATOL_ACC = 5e-1
RTOL_ACC = 1e-5


@pytest.mark.parametrize(
    'functional_mode, quant_sdpa, expected_functions',
    [
        (None, None, set()),
        (
            'input',
            None, {
                torch.nn.functional.linear,
                torch.matmul,
                torch.Tensor.matmul,
                torch.Tensor.__matmul__,
                torch.bmm}),
        (
            'weight',
            None, {
                torch.nn.functional.linear,
                torch.matmul,
                torch.Tensor.matmul,
                torch.Tensor.__matmul__,
                torch.bmm}),
        (
            'all',
            None, {
                torch.nn.functional.linear,
                torch.matmul,
                torch.Tensor.matmul,
                torch.Tensor.__matmul__,
                torch.bmm}),
        (None, 'functional', {torch.nn.functional.scaled_dot_product_attention}),
        (
            'all',
            'functional',
            {
                torch.nn.functional.linear,
                torch.matmul,
                torch.Tensor.matmul,
                torch.Tensor.__matmul__,
                torch.bmm,
                torch.nn.functional.scaled_dot_product_attention}),])
def test_functional_quant_map_modes(functional_mode, quant_sdpa, expected_functions):
    """Functional operand modes and functional SDPA are independently selectable."""
    input_quant = object()
    weight_quant = object()
    quant_map = _functional_quant_map({
        'linear_input_quant': input_quant,
        'weight_quant': weight_quant,
        'q_scaled_quant': 'q',
        'k_transposed_quant': 'k',
        'v_quant': 'v'},
                                      functional_mode,
                                      quant_sdpa)
    assert set(quant_map) == expected_functions
    if functional_mode == 'input':
        assert quant_map[torch.nn.functional.linear] is input_quant
        assert quant_map[torch.matmul] is input_quant
    elif functional_mode == 'weight':
        assert quant_map[torch.matmul] == (None, None, weight_quant)
    elif functional_mode == 'all':
        assert quant_map[torch.matmul] == (input_quant, input_quant, weight_quant)


def test_functional_sdpa_map_does_not_require_linear_quantizers():
    """SDPA-only map construction is independent from functional linear quantizers."""
    quant_map = _functional_quant_map(
        {'q_scaled_quant': 'q', 'k_transposed_quant': 'k', 'v_quant': 'v'},
        quant_sdpa='functional')
    assert quant_map == {
        torch.nn.functional.scaled_dot_product_attention: ('q', 'k', 'v')}


def test_custom_quantizer_can_override_functional_map():
    """Custom quantizers can specialize functional specs independently of layer maps."""
    class FunctionalMapAdjuster(BaseQuantizer):

        @classmethod
        def override_functional_quant_map(cls, quant_map, quantizers_dict):
            quant_map = dict(quant_map)
            quant_map['custom'] = quantizers_dict['weight_quant']
            return quant_map

    weight_quant = object()
    quant_map = FunctionalMapAdjuster.override_functional_quant_map(
        {}, {'weight_quant': weight_quant})
    assert quant_map == {'custom': weight_quant}


def test_functional_weight_mode_does_not_require_input_quantization():
    """Weight-only functional quantization is valid without an input quantizer."""
    args = get_default_args(create_args_parser())
    args.functional_quantization = 'weight'
    args.input_bit_width = None
    validate(args)


@pytest.mark.parametrize('mode', ['input', 'all'])
def test_functional_input_modes_require_input_quantization(mode):
    """Functional input modes fail clearly when no input quantizer is configured."""
    args = get_default_args(create_args_parser())
    args.functional_quantization = mode
    args.input_bit_width = None
    with pytest.raises(AssertionError, match='requires input quantization'):
        validate(args)


def mock_load_raw_dataset(dataset_name: str, split: str, seed: int = 42) -> Dataset:
    assert dataset_name == "c4", f"Expected dataset_name to be c4 but got {dataset_name} instead"
    assert split in ["train", "validation"], f"Expected split to be 'train' or 'validation' but got "
    # Contains information from allenai/c4 (https://huggingface.co/datasets/allenai/c4) which is made available under the ODC Attribution License.
    C4_TEXTS = [
        "Luxembourg's professional networking group for women will host a discussion about promoting Luxembourg abroad.\n(JB) Luxembourg's female only professional networking group will host a discussion about promoting Luxembourg abroad.\nSpeaker Carole Tompers, who is responsible for promoting the Made in Luxembourg products and services to foreign markets, will take guests on a whistle stop tour of the country's key assets.\nHer speech will explore the Nations Brand Index 2010, delve into what makes a Luxembourg brand and suggest ways of strengthening and promoting existing brands abroad.\nMs Tompers has a strong track record in marketing and communications. She currently serves as Secretary General at Luxembourg for Business.\nShe has previously worked on promotional projects with various ministries, the Chamber of Commerce, the Office Ducroire, the National Credit and Investment Corporation, the Chamber of Crafts and Luxembourg's Business Federation.\nThe event is organised by the Network at the Sofitel Kirchberg on November 16, from 7.30pm."
    ]
    return Dataset.from_dict({
        "text": C4_TEXTS,})


def mock_compute_rotations(
    model: nn.Module,
    regions: List[Region],
    full_rotation_method='had',
    fuse_rotations: bool = True,
    expansion_step: int = 1,
    rotation_block_size: Optional[int] = None,
    disable_block_rotation_for_fused: bool = False,
    generator: Optional[torch.Generator] = None,
):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return _compute_rotations(
        model=model,
        regions=regions,
        full_rotation_method=full_rotation_method,
        fuse_rotations=fuse_rotations,
        expansion_step=expansion_step,
        rotation_block_size=rotation_block_size,
        disable_block_rotation_for_fused=disable_block_rotation_for_fused,
        generator=generator)


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


@pytest.mark.llm
def test_custom_quantizer_post_process(caplog, default_run_args, main):
    caplog.set_level(logging.INFO)

    @Registry.register(QUANTIZERS_REGISTRY, "example_inline_model_adjuster")
    class ExampleInlineModelAdjuster(BaseQuantizer):

        @classmethod
        def post_process_quant_model(cls, model: nn.Module) -> nn.Module:
            model.example_inline_model_adjuster_applied = True
            return model

    args = default_run_args
    args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    args.custom_quantizer = "example_inline_model_adjuster"

    _, model = main(args)

    assert getattr(model, "example_inline_model_adjuster_applied", False)


@pytest.mark.llm
def test_custom_quantizer_file_override_and_post_process(caplog, default_run_args, main):
    caplog.set_level(logging.INFO)

    args = default_run_args
    args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    args.custom_quantizer = (
        "tests/brevitas_examples/llm_test_plugin.py:example_quant_and_model_adjuster")

    _, model = main(args)

    assert getattr(model, "example_quant_and_model_adjuster_applied", False)

    weight_proxies = []
    for module in model.modules():
        if hasattr(module, 'weight_quant') and module.weight_quant is not None:
            weight_proxies.append(module.weight_quant)

    for m in model.model.layers:
        # Check input_quant are tied
        assert id(m.self_attn.q_proj.input_quant) == id(m.self_attn.k_proj.input_quant) == id(
            m.self_attn.v_proj.input_quant)
        assert id(m.mlp.up_proj.input_quant) == id(m.mlp.gate_proj.input_quant)

        # Check weight_quant are tied
        assert id(m.self_attn.q_proj.weight_quant) == id(m.self_attn.k_proj.weight_quant) == id(
            m.self_attn.v_proj.weight_quant)
        assert id(m.mlp.up_proj.weight_quant) == id(m.mlp.gate_proj.weight_quant)

    assert weight_proxies
    assert any(
        hasattr(proxy, 'bit_width') and proxy.bit_width() is not None and
        proxy.bit_width().item() == 4 for proxy in weight_proxies)


# Plugin spec for the example ``two_group_optimizer_trainer`` shipped with the
# tests. Importing it (via parse_custom_trainer) registers the trainer into
# TRAINER_REGISTRY as a side-effect.
_LLM_PLUGIN_SPEC = "tests/brevitas_examples/llm_test_plugin.py:two_group_optimizer_trainer"


class _TwoGroupToyModel(nn.Module):
    """Minimal model exposing both a ``q_proj`` submodule and a non-``q_proj``
    submodule, so both parameter selectors of the two-group trainer return
    non-empty parameter lists."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)  # matched by _select_q_proj_params
        self.other_proj = nn.Linear(4, 4)  # matched by _select_non_q_proj_params

    def forward(self, x):
        return self.other_proj(self.q_proj(x))


@pytest.mark.llm
class TestTwoGroupOptimizerTrainer:
    """Tests for the example ``two_group_optimizer_trainer`` custom trainer.

    It registers a ``GeneralizedTrainer`` subclass whose single optimizer
    (AdamW) handles two parameter groups: ``q_proj`` params at ``q_proj_lr``
    and everything else at ``non_q_proj_lr``.
    """

    @pytest.fixture(autouse=True)
    def _load_plugin(self):
        # No skip: if the plugin file is missing, parse_custom_trainer raises
        # FileNotFoundError and the test (correctly) fails.
        parse_custom_trainer(_LLM_PLUGIN_SPEC)

    def _trainer_cls(self):
        return TRAINER_REGISTRY.get("two_group_optimizer_trainer")

    def test_registered(self):
        assert "two_group_optimizer_trainer" in TRAINER_REGISTRY.get_registered_keys()
        trainer_cls = TRAINER_REGISTRY.get("two_group_optimizer_trainer")
        assert issubclass(trainer_cls, GeneralizedTrainer)

    def test_default_lr_values(self):
        args = parse_rotation_optimization_args(
            extra_args=["--max_steps", "1"], trainer_cls=self._trainer_cls())
        assert args.q_proj_lr == pytest.approx(1e-3)
        assert args.non_q_proj_lr == pytest.approx(1e-2)

        os_args = args.optimizer_scheduler_args
        # A single optimizer with two parameter groups.
        assert len(os_args) == 1
        param_setup = os_args[0]["param_setup"]
        assert len(param_setup) == 2
        assert param_setup[0]["optimizer_kwargs"]["lr"] == pytest.approx(1e-3)
        assert param_setup[1]["optimizer_kwargs"]["lr"] == pytest.approx(1e-2)

    def test_cli_override_lr_values(self):
        extra = [
            "--max_steps",
            "1",
            "--q-proj-lr",
            "5e-4",
            "--non-q-proj-lr",
            "5e-2",]
        args = parse_rotation_optimization_args(extra_args=extra, trainer_cls=self._trainer_cls())
        assert args.q_proj_lr == pytest.approx(5e-4)
        assert args.non_q_proj_lr == pytest.approx(5e-2)

        param_setup = args.optimizer_scheduler_args[0]["param_setup"]
        assert param_setup[0]["optimizer_kwargs"]["lr"] == pytest.approx(5e-4)
        assert param_setup[1]["optimizer_kwargs"]["lr"] == pytest.approx(5e-2)

    def test_builds_single_optimizer_two_groups(self):
        args = parse_rotation_optimization_args(
            extra_args=["--max_steps", "1"], trainer_cls=self._trainer_cls())

        model = _TwoGroupToyModel()
        optimizer, _ = _build_optimizers_from_configs(model, args)

        # A single AdamW optimizer (not a MultiOptimizer) with two param groups.
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-2)

        q_ids = {id(p) for name, p in model.named_parameters() if "q_proj" in name}
        group0_params = optimizer.param_groups[0]["params"]
        group1_params = optimizer.param_groups[1]["params"]
        # Each Linear contributes weight + bias.
        assert len(group0_params) == 2
        assert len(group1_params) == 2
        assert all(id(p) in q_ids for p in group0_params)
        assert all(id(p) not in q_ids for p in group1_params)


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
    from brevitas.nn import QuantIdentity
    from brevitas.nn import QuantScaledDotProductAttention as QuantSDPA
    from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
    caplog.set_level(logging.INFO)
    args, _, _ = layer_args_hyperparam

    use_fx = fx_required(args) or args.rotation == 'fused_no_fx'

    if platform.system() == 'Windows' and use_fx:
        pytest.skip("Skipping dynamo + Windows")

    _, model = main(args)
    if args.quant_sdpa == "functional":
        assert hasattr(model, '_functional_quantizers')
        fq_quantizers = list(model._functional_quantizers.items())
        assert len(fq_quantizers) > 0, "Expected functional QuantIdentity quantizers"
        for name, quant_id in fq_quantizers:
            if '_arg1' in name or '_arg2' in name:
                assert isinstance(quant_id, QuantIdentity)
                assert not quant_id.act_quant.is_signed
                assert isinstance(quant_id.act_quant, GroupwiseActQuantProxyFromInjector)
                assert quant_id.act_quant.group_size == args.input_group_size
    else:
        quant_sdpa = [m for m in model.modules() if isinstance(m, QuantSDPA)]
        first_sdpa = quant_sdpa[0]
        assert first_sdpa.q_scaled_quant.act_quant.fused_activation_quant_proxy is None
        assert first_sdpa.attn_output_weights_quant.act_quant.fused_activation_quant_proxy is None
        assert not first_sdpa.v_quant.act_quant.is_signed
        assert not first_sdpa.k_transposed_quant.act_quant.is_signed
        assert isinstance(first_sdpa.v_quant.act_quant, GroupwiseActQuantProxyFromInjector)
        assert isinstance(
            first_sdpa.k_transposed_quant.act_quant, GroupwiseActQuantProxyFromInjector)
        assert first_sdpa.v_quant.act_quant.group_size == args.input_group_size
        assert first_sdpa.k_transposed_quant.act_quant.group_size == args.input_group_size
        if args.quant_sdpa == "fx" or args.quant_sdpa == "eager":
            assert len(quant_sdpa) == 2


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


@pytest.mark.onnx_export
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
    ids=["lighteval", "lighteval_rotations", "lm_eval", "lm_eval_rotations"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "few_shot_eval": "lighteval",
            "few_shot_override_batch_size": 16,
            "few_shot_limit": 16,
            "few_shot_tasks": [
                "arc:challenge|0",
                "winogrande|0",
                "arc:easy|0",
                "hellaswag|0",],
            "few_shot_zeroshot": True,
            "imports": ["lighteval"],
            "all_acc": 0.375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "rotation": "fused_no_fx",
            "replace_rmsnorm": True,
            "eval": False,
            "few_shot_eval": "lighteval",
            "few_shot_override_batch_size": 16,
            "few_shot_limit": 16,
            "few_shot_tasks": [
                "arc:challenge|0",
                "winogrande|0",
                "arc:easy|0",
                "hellaswag|0",],
            "few_shot_zeroshot": True,
            "imports": ["lighteval"],
            "all_acc": 0.375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "eval": False,
            "few_shot_eval": "lm_eval",
            "few_shot_override_batch_size": 16,
            "few_shot_limit": 16,
            "few_shot_tasks": ["arc_challenge", "winogrande", "piqa", "hellaswag"],
            "few_shot_zeroshot": True,
            "imports": ["lm_eval"],
            "all_acc": 0.375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "no_quantize": True,
            "rotation": "fused_no_fx",
            "replace_rmsnorm": True,
            "eval": False,
            "few_shot_eval": "lm_eval",
            "few_shot_override_batch_size": 16,
            "few_shot_limit": 16,
            "few_shot_tasks": ["arc_challenge", "winogrande", "piqa", "hellaswag"],
            "few_shot_zeroshot": True,
            "imports": ["lm_eval"],
            "all_acc": 0.375,},])
def few_shot_eval_args(default_run_args, request):
    # Skip cases for which the LM evaluation library has not been installed
    for lib in request.param["imports"]:
        pytest.importorskip(lib, reason=f"`{lib}` needs to be installed.")
    del request.param["imports"]

    yield process_args_and_metrics(
        default_run_args, request.param, extra_keys=["imports", "all_acc"])


@pytest.mark.few_shot
def test_few_shot_eval(caplog, few_shot_eval_args, main):
    caplog.set_level(logging.INFO)
    args, _, exp_metrics = few_shot_eval_args

    results, _ = main(args)

    # Verify that LM eval metrics match. `strict` is set to False, as
    # only a subset of metrics are checked.
    assert_metrics(results, exp_metrics, atol=ATOL_ACC, rtol=RTOL_ACC, strict=False)


@pytest.mark.llm
@requires_pt_ge('2.4')
def test_rmsnorm_patch_context_manager(caplog):
    """Test that rmsnorm_patch correctly replaces RMSNorm modules on enter
    and restores original modules on exit."""
    from transformers import AutoModelForCausalLM

    caplog.set_level(logging.INFO)

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    config = model.config

    # Discover what RMSNorm classes the model uses before patching
    original_rmsnorm_classes = tuple(
        set(type(m) for m in model.modules() if 'RMS' in type(m).__name__))
    assert len(original_rmsnorm_classes) > 0, "Model should contain at least one RMSNorm class"

    # Collect original module types for all RMSNorm layers before the context manager
    rmsnorm_modules_before = {
        name: type(m) for name, m in model.named_modules() if 'RMS' in type(m).__name__}
    for name, cls in rmsnorm_modules_before.items():
        assert cls in original_rmsnorm_classes, (
            f"Before context manager: {name} should be an original RMSNorm type, got {cls}")
        assert cls is not torch.nn.RMSNorm, (
            f"Before context manager: {name} should not be torch.nn.RMSNorm")

    # Enter the context manager and check that modules are replaced with torch.nn.RMSNorm
    patcher = rmsnorm_patch(model, config, enabled=True)
    patcher.__enter__()
    model_during = patcher.model

    rmsnorm_modules_during = {
        name: type(m) for name, m in model_during.named_modules() if 'RMS' in type(m).__name__}
    assert len(rmsnorm_modules_during) == len(rmsnorm_modules_before), (
        "Number of RMSNorm modules should be the same during the context manager")
    for name, cls in rmsnorm_modules_during.items():
        assert cls is torch.nn.RMSNorm, (
            f"During context manager: {name} should be torch.nn.RMSNorm, got {cls}")

    # Exit the context manager and check that original modules are restored
    patcher.__exit__(None, None, None)
    model_after = patcher.model

    rmsnorm_modules_after = {
        name: type(m) for name, m in model_after.named_modules() if 'RMS' in type(m).__name__}
    assert len(rmsnorm_modules_after) == len(rmsnorm_modules_before), (
        "Number of RMSNorm modules should be the same after the context manager")
    for name, cls in rmsnorm_modules_after.items():
        assert cls in original_rmsnorm_classes, (
            f"After context manager: {name} should be restored to original type, got {cls}")
        assert cls is not torch.nn.RMSNorm, (
            f"After context manager: {name} should not be torch.nn.RMSNorm")
