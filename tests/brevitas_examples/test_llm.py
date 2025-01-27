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

import numpy as np
import onnx
from packaging import version
import pytest
import pytest_cases
import torch
import transformers

from brevitas import config
from brevitas import torch_version
from brevitas_examples.llm.main import main
from brevitas_examples.llm.main import parse_args
from brevitas_examples.llm.main import quantize_llm
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

ATOL_PPL = 2e+02
RTOL_PPL = 1e-04


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


def allclose(x, y, rtol=RTOL_PPL, atol=ATOL_PPL):
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


def transformers_version_ge(required_version: str):
    return version.parse(required_version) >= version.parse(transformers.__version__)


# Check that all args in args are used
def validate_args(args):
    a = vars(args)
    da = vars(parse_args([])[0])
    for k in a.keys():
        assert k in da.keys(), f"Key {k} does not seem to be a valid argument for `quantize_llm`"


def validate_args_and_run_main(args, extra_args=None):
    validate_args(args)
    float_ppl, quant_ppl, model = quantize_llm(args, extra_args=extra_args)
    return float_ppl, quant_ppl, model


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


def requires_fx(args):
    return args.act_equalization == "fx" or args.weight_equalization or args.ln_affine_merge


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
    ],
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
    ])
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
    args.no_float16 = True
    return args


def run_test_models_run_args(args, model_with_ppl):
    args.model = model_with_ppl.name
    exp_float_ppl = model_with_ppl.float_ppl
    use_fx = requires_fx(args)
    if use_fx and not model_with_ppl.supports_fx:
        pytest.xfail(f"{model_with_ppl.name} does not support FX")
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)


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
        "ln_affine_merge=True",],
    params=[
        {},
        {"weight_param_method": "hqo"},
        {"weight_param_method": "hqo", "weight_quant_type": "asym"},
        {"bias_corr": True},
        {"act_equalization": "layerwise"},
        {"act_equalization": "fx"},
        {"weight_equalization": True},
        {"gptq": True},
        {"ln_affine_merge": True},])
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
    scope="session",
    ids=[
        "opt",],
    params=[
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            float_ppl=None,
            supports_fx=True,
        ),])
def small_models_with_ppl_pt_ge_2_4(request):
    yield request.param


@pytest.mark.llm
@requires_pt_ge('2.4')
def test_small_models_toggle_run_args_pt_ge_2_4(
        caplog, toggle_run_args, small_models_with_ppl_pt_ge_2_4):
    caplog.set_level(logging.INFO)
    run_test_models_run_args(toggle_run_args, small_models_with_ppl_pt_ge_2_4)


@pytest_cases.fixture(
    ids=[
        "llama",
        "mistral",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "fx",
            "bias_corr": True,
            "float_ppl": 33312.0 if transformers_version_ge('4.46.0') else 33239.5,
            "quant_ppl": 33056.0 if transformers_version_ge('4.46.0') else 33283.75390625},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_equalization": "layerwise",
            "gptq": True,
            "float_ppl": 31056.0 if transformers_version_ge('4.46.0') else 31274.05078125,
            "quant_ppl": 33056.0 if transformers_version_ge('4.46.0') else 33139.23046875},])
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
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest_cases.fixture(
    ids=[
        "opt-replace-mha",
        "opt-quant-sdpa",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "weight_equalization": True,
            "ln_affine_merge": True,
            "replace_mha": True,
            "float_ppl": 50016.0,
            "quant_ppl": 50016.0},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
            "weight_equalization": True,
            "ln_affine_merge": True,
            "quant_sdpa": True,
            "float_ppl": 50016.0,
            "quant_ppl": 50016.0},])
def acc_args_and_acc_pt_ge_2_4(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    args.update(**run_dict)
    yield args, float_ppl, quant_ppl


@pytest.mark.llm
@requires_pt_ge('2.4')
def test_small_models_acc_pt_ge_2_4(caplog, acc_args_and_acc_pt_ge_2_4):
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = acc_args_and_acc_pt_ge_2_4
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
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
        "mistral-int8-quant-last-layer",],
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
                "lm_head": "<class 'brevitas.nn.quant_linear.QuantLinear'>"},},
    ])  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
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
        if torch_version < version.parse('2.4'):
            pytest.skip("Replacing RMSNorm requires torch 2.4+ or greater")
        if hasattr(args, 'rotation') and args.rotation == 'fx' and platform.system() == 'Windows':
            pytest.skip("Skipping dynamo + windows")
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
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
        "llama-rotation-full-fx-sdpa"],
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
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 2,  # Sinks: Only Down proj
                "<class 'torch.nn.modules.linear.Linear'>":
                    15,  # LM Head + Q/K/V/O projs + Up/Gate/Down projs
                "<class 'torch.nn.modules.normalization.RMSNorm'>": 5,
                "<class 'torch.nn.modules.normalization.LayerNorm'>": 0,}},])
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
        if torch_version < version.parse('2.4'):
            pytest.skip("Replacing RMSNorm requires torch 2.4+ or greater")
        if hasattr(args, 'rotation') and args.rotation == 'fx' and platform.system() == 'Windows':
            pytest.skip("Skipping dynamo + windows")
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    assert_layer_types_count(model, exp_layer_types_count)


@pytest_cases.fixture(
    ids=[
        "opt-replace-mha",
        "opt-quant-sdpa",],
    params=[
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
def layer_args_pt_ge_2_4(default_run_args, request):
    args = default_run_args
    layer_dict = request.param
    exp_layer_types = layer_dict["exp_layer_types"]
    del layer_dict["exp_layer_types"]
    args.update(**layer_dict)
    yield args, exp_layer_types


@pytest.mark.llm
@requires_pt_ge('2.4')
def test_small_models_quant_layer_pt_ge_2_4(caplog, layer_args_pt_ge_2_4):
    caplog.set_level(logging.INFO)
    args, exp_layer_types = layer_args_pt_ge_2_4
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    assert_layer_types(model, exp_layer_types)


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
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    onnx_model = onnx.load(os.path.join(args.export_prefix, "model.onnx"))
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
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    filepath = args.export_prefix + ".pt"
    torchscript_model = torch.jit.load(filepath)
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
            "float_ppl": 33238.8984375,
            "quant_ppl": 33252.21484375},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "learned_round": "linear_round",
            "learned_round_iters": 1,
            "gpxq_block_name": "model.layers",
            "float_ppl": 31275.958984375,
            "quant_ppl": 31337.4921875},])
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
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
    assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest_cases.fixture(
    ids=[
        "llama_fused_rotation_ort",
        "llama_fused_rotation_ort_no_orphan",
        "llama_fused_rotation_had",
        "llama_fused_rotation_had_no_orphan",
        "llama_layerwise",],
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
            "float_ppl": 33238.8984375,
            "quant_ppl": 33232.65234375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": False,
            "rotation_mode": "ort",
            "float_ppl": 33238.8984375,
            "quant_ppl": 33420.65234375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": True,
            "rotation_mode": "had",
            "float_ppl": 33238.8984375,
            "quant_ppl": 33290.48046875,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "fused_no_fx",
            "rotation_orphan_sink": False,
            "rotation_mode": "had",
            "float_ppl": 33238.8984375,
            "quant_ppl": 33204.80859375,},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_calibration": False,
            "weight_bit_width": 4,
            "input_bit_width": None,
            "replace_rmsnorm": True,
            "rotation": "layerwise",
            "float_ppl": 33238.8984375,
            "quant_ppl": 33446.734375,},])
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
    float_ppl, quant_ppl, model = validate_args_and_run_main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
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
            "no_float16": True,
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 33238.8984375,
            "quant_ppl": 33239.33984375,
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
            "no_float16": True,
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 33238.8984375,
            "quant_ppl": 33423.0390625,
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
            "no_float16": True,
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 33238.8984375,
            "quant_ppl": 33286.98828125,
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
            "no_float16": True,
            "extra_args": [
                "--learning_rate",
                "1.5",
                "--max_steps",
                "2",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "1"],
            "float_ppl": 33238.8984375,
            "quant_ppl": 33175.3046875,
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
    float_ppl, quant_ppl, _ = validate_args_and_run_main(args, extra_args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
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
    with patch('brevitas_examples.llm.main.fuse_parametrized_rotations', lambda model: model):
        _, _, model = validate_args_and_run_main(args, extra_args)
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
