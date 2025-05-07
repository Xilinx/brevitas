# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import copy
from dataclasses import dataclass
import logging
import os
import platform
import shutil
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import onnx
from packaging import version
import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas import torch_version
from brevitas_examples.llm.main import fx_required
from brevitas_examples.llm.main import main
from brevitas_examples.llm.main import parse_args
from brevitas_examples.llm.main import quantize_llm
from tests.marker import jit_disabled_for_dynamic_quant_act
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

ATOL_PPL = 1e+01
RTOL_PPL = 1e-04

MODEL_PT_VERSION_REQUIREMENTS = {
    "hf-internal-testing/tiny-random-LlamaForCausalLM": "2.0",
    "hf-internal-testing/tiny-random-MistralForCausalLM": "2.0",
    "hf-internal-testing/tiny-random-OPTForCausalLM": "2.4",}


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


def allclose(x, y, rtol=RTOL_PPL, atol=ATOL_PPL):
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


# Check that all args in args are used
def validate_args(args):
    a = vars(args)
    da = vars(parse_args([])[0])
    for k in a.keys():
        assert k in da.keys(), f"Key {k} does not seem to be a valid argument for `quantize_llm`"
    req_pt = MODEL_PT_VERSION_REQUIREMENTS[args.model]
    if torch_version < version.parse(req_pt):
        pytest.skip(f"{args.model} requires PyTorch version {req_pt}")
    if args.replace_rmsnorm:
        if torch_version < version.parse('2.4'):
            pytest.skip("Replacing RMSNorm requires torch 2.4+ or greater")
    if args.gpxq_block_name == "model.layers" and args.learned_round is not None and "opt" in args.model.lower(
    ):
        pytest.skip(
            f"OPT-style model {args.model} not support with learned_round={args.learned_round} with block module named {args.gpxq_block_name}"
        )


def validate_args_and_run_main(args, extra_args=None):
    validate_args(args)
    with patch('brevitas_examples.llm.llm_quant.data_utils.load_raw_dataset',
               mock_load_raw_dataset):
        results, model = quantize_llm(args, extra_args=extra_args)
    return results, model


def assert_layer_types(model, exp_layer_types):
    for key, string in exp_layer_types.items():
        matched = False
        layer_names = []
        for name, layer in model.named_modules():
            layer_names += [name]
            if name == key:
                matched = True
                ltype = str(type(layer))
                assert ltype == string, f"Expected layer type: {string}, found {ltype} for key: {key}"
                continue
        assert matched, f"Layer key: {key} not found in {layer_names}"


def assert_layer_types_count(model, exp_layer_types_count):
    layer_types_count = {}
    for name, layer in model.named_modules():
        ltype = str(type(layer))
        if ltype not in layer_types_count:
            layer_types_count[ltype] = 0
        layer_types_count[ltype] += 1

    for name, count in exp_layer_types_count.items():
        curr_count = 0 if name not in layer_types_count else layer_types_count[name]
        assert count == curr_count, f"Expected {count} instances of layer type: {name}, found {curr_count}."


class UpdatableNamespace(Namespace):

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)


@dataclass
class ModelAndPpl:
    name: str
    float_ppl: float
    supports_fx: bool


@pytest_cases.fixture(
    scope="session",
    ids=[
        "llama",
        "mistral",  #"mixtral",
        "opt",],
    params=[
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            float_ppl=None,
            supports_fx=True,
        ),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-MistralForCausalLM",
            float_ppl=None,
            supports_fx=False,
        ),
        #ModelAndPpl( # Ready for MoE support
        #    name="dacorvo/Mixtral-tiny",
        #    float_ppl=None,
        #    supports_fx=True,
        #),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            float_ppl=None,
            supports_fx=True,
        ),])
def small_models_with_ppl(request):
    yield request.param


@pytest_cases.fixture()
def default_run_args(request):
    args = UpdatableNamespace(**vars(parse_args([])[0]))
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


def run_test_models_run_args(args, model_with_ppl):
    args.model = model_with_ppl.name
    use_fx = fx_required(args)
    if use_fx and not model_with_ppl.supports_fx:
        pytest.xfail(f"{model_with_ppl.name} does not support FX")
    if args.input_scale_type == 'dynamic' and config.JIT_ENABLED:
        pytest.skip("Dynamic activation not compatible with JIT")
    if platform.system() == 'Windows' and hasattr(args, 'rotation') and args.rotation in [
            'fx', 'fused_no_fx']:
        pytest.skip("Skipping dynamo + Windows")

    validate_args_and_run_main(args)


# yapf: disable
@pytest_cases.fixture(
    ids=[
        "defaults",
        "sym_weight_param_method=hqo",
        "asym_weight_param_method=hqo",
        "bias_corr=True",
        "act_equalization=layerwise",
        "act_equalization=fx",
        "weight_equalization=True",
        "gptq=True",
        "ln_affine_merge=True",
        "rotation=layerwise",
        "rotation=fx",
        "rotation=fused_no_fx",
        "act_equalization=fx,gptq=True",
        "quant_sdpa_fx_per_row",
        "quant_sdpa_functional_per_row",
        "functional_sdpa_quant=True,rotation=fused_no_fx",
        "per_group_w_padding,learned_round=linear_round",],
    params=[
        {},
        {"weight_param_method": "hqo"},
        {"weight_param_method": "hqo", "weight_quant_type": "asym"},
        {"bias_corr": True},
        {"act_equalization": "layerwise"},
        {"act_equalization": "fx"},
        {"weight_equalization": True},
        {"gptq": True},
        {"ln_affine_merge": True},
        {"rotation": "layerwise"},
        {"rotation": "fx", "ln_affine_merge": True, "replace_rmsnorm": True, "convert_layernorm_to_rmsnorm": True},
        {"rotation": "fused_no_fx", "replace_rmsnorm": True},
        {"act_equalization": "fx", "gptq": True},
        {"quant_sdpa": True, "input_scale_type": "dynamic", "input_quant_granularity": "per_row"},
        {"functional_sdpa_quant": True, "input_scale_type": "dynamic", "input_quant_granularity": "per_row"},
        {
            "functional_sdpa_quant": True,
            "rotation": "fused_no_fx",
            "rotation_sdpa_regions": True,
            "input_scale_type": "dynamic",
            "replace_rmsnorm": True
        }, {
            "weight_quant_granularity": "per_group",
            "weight_group_size": 11,
            "learned_round": "linear_round",
            "learned_round_iters": 1,
            "gpxq_block_name": "model.layers",
        },])
# yapf: enable
def toggle_run_args(default_run_args, request):
    args = default_run_args
    args.update(**request.param)
    if args.weight_param_method == 'hqo' and config.JIT_ENABLED:
        pytest.skip("Local loss mode requires JIT to be disabled")
    yield args


@pytest.mark.llm
@requires_pt_ge('2.2')
def test_small_models_toggle_run_args(caplog, toggle_run_args, small_models_with_ppl):
    caplog.set_level(logging.INFO)
    run_test_models_run_args(toggle_run_args, small_models_with_ppl)


@pytest_cases.fixture(
    ids=[
        "llama",
        "llama_float_dynamic_input",
        "mistral",
        "opt-replace-mha",
        "opt-quant-sdpa",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "fx",
            "bias_corr": True,
            "float_ppl": 32428.475,
            "quant_ppl": 32327.721},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "fx",
            "bias_corr": True,
            "weight_quant_format": "float_ocp_e4m3",
            "input_quant_format": "float_ocp_e4m3",
            "input_quant_granularity": "per_row",
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "float_ppl": 32428.475,
            "quant_ppl": 32428.383},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_equalization": "layerwise",
            "gptq": True,
            "float_ppl": 36796.984,
            "quant_ppl": 36910.191},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "weight_equalization": True,
            "ln_affine_merge": True,
            "replace_mha": True,
            "float_ppl": 51649.797,
            "quant_ppl": 51694.785},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "weight_equalization": True,
            "ln_affine_merge": True,
            "quant_sdpa": True,
            "float_ppl": 51649.797,
            "quant_ppl": 51688.922},])
def acc_args_and_acc(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    args.update(**run_dict)
    yield args, float_ppl, quant_ppl


@pytest.mark.llm
@requires_pt_ge('2.2')
def test_small_models_acc(caplog, acc_args_and_acc):
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = acc_args_and_acc
    if args.input_scale_type == 'dynamic' and config.JIT_ENABLED:
        pytest.skip("Dynamic activation not compatible with JIT")
    results, _ = validate_args_and_run_main(args)
    float_ppl = results["float_ppl"].detach().cpu().numpy()
    quant_ppl = results["quant_ppl"].detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest_cases.fixture(
    ids=[
        "mistral-int8",
        "mistral-weight-only",
        "mistral-fp8_ocp",
        "mistral-fp8_fnuz",
        "llama-mxfp8",
        "llama-int8-act_equalization=layerwise",
        "mistral-int8-quant-last-layer",
        "llama-int8-svd_quant",
        "opt-replace-mha",
        "opt-quant-sdpa",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "exp_layer_types": {
                "lm_head":
                    "<class 'torch.nn.modules.linear.Linear'>",
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant":
                    "<class 'brevitas.core.quant.int.RescalingIntQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.int.RescalingIntQuant'>",},
        },  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "input_bit_width": None,
            "act_calibration": False,
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant":
                    "<class 'brevitas.proxy.runtime_quant.ActQuantProxyFromInjector'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.int.RescalingIntQuant'>",},
        },  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_quant_type": "sym",
            "input_quant_format": "float_ocp_e5m2",
            "input_quant_type": "sym",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",},
        },  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "weight_quant_format": "float_fnuz_e4m3",
            "weight_quant_type": "sym",
            "input_quant_format": "float_fnuz_e5m2",
            "input_quant_type": "sym",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",},
        },  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_scale_precision": "po2_scale",
            "weight_param_method": "stats",
            "weight_quant_granularity": "per_group",
            "weight_group_size": 16,
            "weight_quant_type": "sym",
            "input_quant_format": "float_ocp_e5m2",
            "input_scale_type": "dynamic",
            "input_scale_precision": "po2_scale",
            "input_param_method": "stats",
            "input_quant_granularity": "per_group",
            "input_group_size": 16,
            "input_quant_type": "sym",
            "act_calibration": False,
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant.input_view_impl":
                    "<class 'brevitas.core.function_wrapper.shape.DynamicOverSubChannelBlockView'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant.input_view_impl":
                    "<class 'brevitas.core.function_wrapper.shape.OverSubChannelBlockView'>",},},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "layerwise",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.equalized_layer.EqualizedModule'>",
                "model.layers.0.self_attn.q_proj.layer":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",},},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "quantize_last_layer": True,
            "exp_layer_types": {
                "lm_head": "<class 'brevitas.nn.quant_linear.QuantLinear'>"},
        },  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "svd_quant": True,
            "svd_quant_rank": 4,
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas_examples.common.svd_quant.ErrorCorrectedModule'>",
                "model.layers.0.self_attn.q_proj.layer":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",},},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "replace_mha": True,
            "exp_layer_types": {
                "model.decoder.layers.0.self_attn":
                    "<class 'brevitas_examples.llm.llm_quant.mha_layers.QuantizableOPTAttention'>",
                "model.decoder.layers.0.self_attn.mha":
                    "<class 'brevitas.nn.quant_mha.QuantMultiheadAttention'>",}},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "quant_sdpa": True,
            "exp_layer_types": {
                "scaled_dot_product_attention":
                    "<class 'brevitas.nn.quant_sdpa.QuantScaledDotProductAttention'>",}},])
def layer_args(default_run_args, request):
    args = default_run_args
    layer_dict = request.param
    exp_layer_types = layer_dict["exp_layer_types"]
    del layer_dict["exp_layer_types"]
    args.update(**layer_dict)
    yield args, exp_layer_types


@pytest.mark.llm
@requires_pt_ge('2.2')
def test_small_models_quant_layer(caplog, layer_args):
    caplog.set_level(logging.INFO)
    args, exp_layer_types = layer_args
    if args.replace_rmsnorm:
        if hasattr(args, 'rotation') and args.rotation == 'fx' and platform.system() == 'Windows':
            pytest.skip("Skipping dynamo + windows")
    _, model = validate_args_and_run_main(args)
    assert_layer_types(model, exp_layer_types)


@pytest_cases.fixture(
    ids=[
        "mistral-int8",
        "mistral-weight-only",
        "mistral-fp8_ocp",
        "mistral-fp8_fnuz",
        "llama-mxfp8",
        "llama-int8-act_equalization=layerwise",
        "mistral-int8-quant-last-layer",
        "llama-rotation-mixed-fx",
        "llama-rotation-full-fx",
        "llama-rotation-full-fx-sdpa",
        "llama-int8-svd_quant"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "exp_layer_types_count": {
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'brevitas.core.quant.int.RescalingIntQuant'>": 28,
            }},  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "input_bit_width": None,
            "act_calibration": False,
            "exp_layer_types_count": {
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'brevitas.core.quant.int.RescalingIntQuant'>": 14,
            }},  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_quant_type": "sym",
            "input_quant_format": "float_ocp_e5m2",
            "input_quant_type": "sym",
            "exp_layer_types_count": {
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'brevitas.core.quant.float.FloatQuant'>": 28,}},  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "weight_quant_format": "float_fnuz_e4m3",
            "weight_quant_type": "sym",
            "input_quant_format": "float_fnuz_e5m2",
            "input_quant_type": "sym",
            "exp_layer_types_count": {
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'brevitas.core.quant.float.FloatQuant'>": 28,}},  # input_quant/weight_quant
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_scale_precision": "po2_scale",
            "weight_param_method": "stats",
            "weight_quant_granularity": "per_group",
            "weight_group_size": 16,
            "weight_quant_type": "sym",
            "input_quant_format": "float_ocp_e5m2",
            "input_scale_type": "dynamic",
            "input_scale_precision": "po2_scale",
            "input_param_method": "stats",
            "input_quant_granularity": "per_group",
            "input_group_size": 16,
            "input_quant_type": "sym",
            "act_calibration": False,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'brevitas.core.quant.float.FloatQuant'>": 28,  # input_quant/weight_quant
                "<class 'brevitas.core.function_wrapper.shape.DynamicOverSubChannelBlockView'>":
                    14,  # input_quant..input_view_impl/input_quant..scaling_impl.input_view_impl
                "<class 'brevitas.core.function_wrapper.shape.OverSubChannelBlockView'>":
                    28,  # weight_quant..input_view_impl/weight_quant..scaling_impl.input_view_impl
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>": 5,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "layerwise",
            "exp_layer_types_count": {
                "<class 'brevitas.nn.quant_linear.QuantLinear'>":
                    14,  # Q/K/V/O projs + Up/Gate/Down projs
                "<class 'torch.nn.modules.linear.Linear'>": 1,  # LM Head
                "<class 'brevitas.nn.equalized_layer.EqualizedModule'>":
                    15,  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
                "<class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>": 5,}},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "quantize_last_layer": True,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.quant_linear.QuantLinear'>": 15,
            }},  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "ln_affine_merge": True,
            "replace_rmsnorm": True,
            "quantize_last_layer": True,
            "no_quantize": True,
            "rotation_orphan_sink": True,
            "convert_layernorm_to_rmsnorm": True,
            "rotation": "fx",
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>":
                    4,  # Sinks: O proj + Down proj
                "<class 'torch.nn.modules.linear.Linear'>":
                    15,  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
                "<class 'torch.nn.modules.normalization.RMSNorm'>": 5,
                "<class 'torch.nn.modules.normalization.LayerNorm'>": 0,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "ln_affine_merge": True,
            "replace_rmsnorm": True,
            "quantize_last_layer": True,
            "no_quantize": True,
            "rotation_orphan_sink": False,
            "convert_layernorm_to_rmsnorm": True,
            "rotation": "fx",
            "exp_layer_types_count": {
                "<class 'torch.nn.modules.linear.Linear'>":
                    15,  # LM Head + Q/K/V projs + Up/Gate/Down projs
                "<class 'torch.nn.modules.normalization.RMSNorm'>": 5,  # Input + Post attention
                "<class 'torch.nn.modules.normalization.LayerNorm'>": 0,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "ln_affine_merge": True,
            "replace_rmsnorm": True,
            "quantize_last_layer": True,
            "no_quantize": True,
            "rotation_orphan_sink": True,
            "convert_layernorm_to_rmsnorm": True,
            "rotation_sdpa_regions": True,
            "rotation": "fx",
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>":
                    4,  # Sinks: Out proj + Down proj
                "<class 'torch.nn.modules.linear.Linear'>":
                    15,  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
                "<class 'torch.nn.modules.normalization.RMSNorm'>": 5,
                "<class 'torch.nn.modules.normalization.LayerNorm'>": 0,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "svd_quant": True,
            "svd_quant_rank": 4,
            "exp_layer_types_count": {
                "<class 'brevitas_examples.common.svd_quant.ErrorCorrectedModule'>": 14,
                "<class 'brevitas.nn.quant_linear.QuantLinear'>": 14,}},])
def layer_args_types_count(default_run_args, request):
    args = default_run_args
    layer_dict = request.param
    exp_layer_types_count = layer_dict["exp_layer_types_count"]
    del layer_dict["exp_layer_types_count"]
    args.update(**layer_dict)
    yield args, exp_layer_types_count


@pytest.mark.llm
@requires_pt_ge('2.2')
def test_small_models_quant_layer_types_count(caplog, layer_args_types_count):
    caplog.set_level(logging.INFO)
    args, exp_layer_types_count = layer_args_types_count
    if args.replace_rmsnorm:
        if hasattr(args, 'rotation') and args.rotation == 'fx' and platform.system() == 'Windows':
            pytest.skip("Skipping dynamo + windows")
    _, model = validate_args_and_run_main(args)
    assert_layer_types_count(model, exp_layer_types_count)


@pytest_cases.fixture(
    ids=["mistral-kv-quant-fx-sdpa", "mistral-kv-quant-functional-sdpa"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "input_quant_granularity": "per_row",
            "kv_quant_granularity": "per_group",
            "input_group_size": 32,
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "quant_sdpa": True,
            "functional_sdpa_quant": False,
            "kv_quant_type": "asym"},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "input_quant_granularity": "per_row",
            "kv_quant_granularity": "per_group",
            "input_group_size": 32,
            "input_scale_type": "dynamic",
            "input_quant_type": "sym",
            "quant_sdpa": False,
            "functional_sdpa_quant": True,
            "kv_quant_type": "asym"},])
def layer_args_hyperparam(default_run_args, request):
    args = default_run_args
    layer_dict = request.param
    args.update(**layer_dict)
    yield args


@pytest.mark.llm
@requires_pt_ge('2.2')
@jit_disabled_for_dynamic_quant_act()
def test_small_models_quant_layer_hyperparam(caplog, layer_args_hyperparam):
    from brevitas.nn import QuantScaledDotProductAttention as QuantSDPA
    from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
    caplog.set_level(logging.INFO)
    args = layer_args_hyperparam

    _, model = validate_args_and_run_main(args)
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
    if args.quant_sdpa:
        assert len(quant_sdpa) > 1
    elif args.functional_sdpa_quant:
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
    args = default_run_args
    export_dict = request.param
    args.update(**export_dict)
    yield args


@pytest.mark.llm
@jit_disabled_for_export()
@requires_pt_ge('2.2')
def test_small_models_onnx_export(caplog, onnx_export_args):
    caplog.set_level(logging.INFO)
    args = onnx_export_args
    validate_args_and_run_main(args)
    onnx.load(os.path.join(args.export_prefix, "model.onnx"))
    shutil.rmtree(args.export_prefix)


@pytest_cases.fixture(
    ids=[
        "qcdq-asym",
        "qcdq-sym",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "eval": False,
            "quantize_weight_zero_point": True,
            "quantize_input_zero_point": True,
            "export_target": "torch_qcdq",},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "eval": False,
            "weight_quant_type": "sym",
            "input_quant_type": "sym",
            "export_target": "torch_qcdq",},])
def torch_export_args(default_run_args, request):
    args = default_run_args
    export_dict = request.param
    args.update(**export_dict)
    yield args


@pytest.mark.llm
@jit_disabled_for_export()
@requires_pt_ge('2.2')
def test_small_models_torch_export(caplog, torch_export_args):
    caplog.set_level(logging.INFO)
    args = torch_export_args
    validate_args_and_run_main(args)
    filepath = args.export_prefix + ".pt"
    torch.jit.load(filepath)
    os.remove(filepath)


@pytest_cases.fixture(
    ids=[
        "llama",
        "mistral",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "learned_round": "linear_round",
            "learned_round_iters": 1,
            "gpxq_block_name": "model.layers",
            "float_ppl": 32428.475,
            "quant_ppl": 32533.578},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "learned_round": "linear_round",
            "learned_round_iters": 1,
            "gpxq_block_name": "model.layers",
            "float_ppl": 36796.984,
            "quant_ppl": 36821.664},])
def learned_round_ppl_args_and_ppl(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    args.update(**run_dict)
    yield args, float_ppl, quant_ppl


@pytest.mark.llm
@requires_pt_ge('2.2')
def test_small_models_learned_round_ppl(caplog, learned_round_ppl_args_and_ppl):
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = learned_round_ppl_args_and_ppl
    results, _ = validate_args_and_run_main(args)
    float_ppl = results["float_ppl"].detach().cpu().numpy()
    quant_ppl = results["quant_ppl"].detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest_cases.fixture(
    ids=[
        "llama_fused_rotation_ort",
        "llama_fused_rotation_ort_no_orphan",
        "llama_fused_rotation_had",
        "llama_fused_rotation_had_no_orphan",
        "llama_layerwise",
        "llama_fused_rotation_had_no_orphan_expanded"],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": True,
            "rotation_mode": "ort",
            "float_ppl": 32428.475,
            "quant_ppl": 32405.289,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": False,
            "rotation_mode": "ort",
            "float_ppl": 32428.475,
            "quant_ppl": 32351.035,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": True,
            "rotation_mode": "had",
            "float_ppl": 32428.475,
            "quant_ppl": 32410.234,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": False,
            "rotation_mode": "had",
            "float_ppl": 32428.475,
            "quant_ppl": 32512.951},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "layerwise",
            "float_ppl": 32428.475,
            "quant_ppl": 32537.238,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": False,
            "rotation_mode": "had",
            "rotation_layers_to_expand": ["down_proj"],
            "float_ppl": 32428.475,
            "quant_ppl": 32515.525,},])
def rotation_ppl_args_and_ppl(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    args.update(**run_dict)
    yield args, float_ppl, quant_ppl


@requires_pt_ge('2.4')
def test_small_models_rotation_ppl(caplog, rotation_ppl_args_and_ppl):
    if platform.system() == "Windows":
        pytest.skip("Skipping dynamo + windows")
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = rotation_ppl_args_and_ppl
    results, _ = validate_args_and_run_main(args)
    float_ppl = results["float_ppl"].detach().cpu().numpy()
    quant_ppl = results["quant_ppl"].detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest_cases.fixture(
    ids=[
        "llama_rotation_optimization_ort",
        "llama_rotation_optimization_ort_no_orphan",
        "llama_rotation_optimization_had",
        "llama_rotation_optimization_had_no_orphan",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "optimize_rotations": True,
            "rotation_orphan_sink": True,
            "rotation_mode": "ort",
            "nsamples_rot_calibration": 2,
            "dtype": "float32",
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 32428.475,
            "quant_ppl": 32414.531,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 4,
                "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "optimize_rotations": True,
            "rotation_orphan_sink": False,
            "rotation_mode": "ort",
            "nsamples_rot_calibration": 2,
            "dtype": "float32",
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 32428.475,
            "quant_ppl": 32342.799,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 0,
                "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "optimize_rotations": True,
            "rotation_orphan_sink": True,
            "rotation_mode": "had",
            "nsamples_rot_calibration": 2,
            "dtype": "float32",
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 32428.475,
            "quant_ppl": 32491.781,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 4,
                "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "optimize_rotations": True,
            "rotation_orphan_sink": False,
            "rotation_mode": "had",
            "nsamples_rot_calibration": 2,
            "dtype": "float32",
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 32428.475,
            "quant_ppl": 32452.111,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 0,
                "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},])
def rotation_optimization_args_layer_count_and_ppl(default_run_args, request):
    args = default_run_args
    run_dict = copy.deepcopy(request.param)
    extra_args = run_dict["extra_args"]
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    exp_layer_types_count = run_dict["exp_layer_types_count"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    del run_dict["extra_args"]
    del run_dict["exp_layer_types_count"]
    args.update(**run_dict)
    yield args, extra_args, float_ppl, quant_ppl, exp_layer_types_count


@requires_pt_ge('2.4')
def test_small_models_rotation_optimization_ppl(
        caplog, rotation_optimization_args_layer_count_and_ppl):
    if platform.system() != "Linux":
        pytest.skip("Skipping dynamo + windows/macos")
    # Tolerances are stricter for this test, to ensure that it does not pass
    # with non-optimized quantized perplexities
    RTOL_ROT, ATOL_ROT = 1e-05, 2.
    caplog.set_level(logging.INFO)
    args, extra_args, exp_float_ppl, exp_quant_ppl, _ = rotation_optimization_args_layer_count_and_ppl
    results, _ = validate_args_and_run_main(args, extra_args)
    float_ppl = results["float_ppl"].detach().cpu().numpy()
    quant_ppl = results["quant_ppl"].detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl, rtol=RTOL_ROT, atol=ATOL_ROT), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@requires_pt_ge('2.4')
def test_small_models_rotation_optimization_layer_count(
        caplog, rotation_optimization_args_layer_count_and_ppl):
    if platform.system() != "Linux":
        pytest.skip("Skipping dynamo + windows/macos")
    # Tolerances are stricter for this test, to ensure that it does not pass
    # with non-optimized quantized perplexities
    caplog.set_level(logging.INFO)
    args, extra_args, _, _, exp_layer_types_count = rotation_optimization_args_layer_count_and_ppl
    with patch('brevitas_examples.llm.main.fuse_parametrizations', lambda model: model):
        _, model = validate_args_and_run_main(args, extra_args)
    assert_layer_types_count(model, exp_layer_types_count)


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
            main()
