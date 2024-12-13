# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import copy
from dataclasses import dataclass
from functools import partial
from itertools import product
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
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from brevitas import config
from brevitas import torch_version
# LLM example depends on optimum-amd, which requires PyTorch>=2.2
from brevitas_examples.llm.main import quantize_llm
from brevitas_examples.llm.main import parse_args

from brevitas.graph.equalize import _apply_had_device
from brevitas.nn.equalized_layer import RotatedModule
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.rotation_utils import extract_trainable_rotation_matrices
from brevitas_examples.llm.llm_quant.rotation_utils import fuse_rotations
from brevitas_examples.llm.main import fused_rotation_no_fx

from tests.conftest import SEED
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

ATOL = 1e-3


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


def allclose(x, y):
    return np.allclose(x, y, rtol=1e-03, atol=1e+01, equal_nan=False)


def allveryclose(x, y):
    return np.allclose(x, y, rtol=1e-04, atol=2e+02, equal_nan=False)


def allexact(x, y):
    return np.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=False)


def transformers_version_ge(required_version: str):
    return version.parse(required_version) >= version.parse(transformers.__version__)


# Check that all args in args are used
def validate_args(args):
    a = vars(args)
    da = vars(parse_args([]))
    for k in a.keys():
        assert k in da.keys(), f"Key {k} does not seem to be a valid argument for `quantize_llm`"


def validate_args_and_run_main(args):
    validate_args(args)
    float_ppl, quant_ppl, model = quantize_llm(args)
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
    assert allveryclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allveryclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


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
    assert allveryclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allveryclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


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
        "llama-rotation-full-fx",],
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
    assert allveryclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allveryclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


# Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/eval_utils/rotation_utils.py#L26
# This functions needs to be patches to enable passing the generator and ensuring that the orthogonal
# matrices generated are the same.
def _random_orthogonal_matrix(size, generator):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0).float()
    return q


@pytest_cases.fixture(
    ids=[
        "llama",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "input_bit_width": None,
            "fuse_sequences": False,
            "act_calibration": False,},])
def equalize_args(default_run_args, request):
    args = default_run_args
    export_dict = request.param
    args.update(**export_dict)
    yield args


# Auxiliar method to compare the weights in rotated modules.
def _compare_fused_unfused_rotation_modules(module_name, fused_rot_module, unfused_rot_module):
    fused_weight = fused_rot_module.weight if isinstance(
        fused_rot_module, nn.Linear) else fused_rot_module.layer.weight
    fused_bias = fused_rot_module.bias if isinstance(
        fused_rot_module, nn.Linear) else fused_rot_module.layer.bias
    unfused_weight = unfused_rot_module.weight if isinstance(
        unfused_rot_module, nn.Linear) else unfused_rot_module.layer.weight
    unfused_bias = unfused_rot_module.bias if isinstance(
        unfused_rot_module, nn.Linear) else unfused_rot_module.layer.bias
    assert torch.allclose(fused_weight, unfused_weight, rtol=0.0, atol=0.0), f"The weights after rotation do not match for module {module_name}."
    if fused_bias is not None:
        assert torch.allclose(fused_bias, unfused_bias, rtol=0.0, atol=0.0), f"The bias after rotation do not match for module {module_name}."
        # In case a RotatedModule is found, additional checks need to be done.
        if isinstance(fused_rot_module, RotatedModule):
            assert isinstance(unfused_rot_module, RotatedModule), f"Expected an instance of RotatedModule for module {module_name}."
            assert torch.allclose(fused_rot_module.had_mat, unfused_rot_module.had_mat, rtol=0.0, atol=0.0), f"The rotation matrices of RotatedModule {module_name} do not match."


@pytest.mark.llm
@requires_pt_ge('2.4')
@pytest_cases.parametrize(
    'partial_had, fused_rotations, add_additional_regions',
    list(product([False, True], repeat=3)),
    ids=[("fused-R1" if fused_rotations else "R1") + ("-R2" if add_additional_regions else "") +
         ("-R3" if partial_had else "") for partial_had,
         fused_rotations,
         add_additional_regions in list(product([False, True], repeat=3))],
)
@pytest_cases.parametrize('rotation_mode', ['ort', 'had'])
def test_small_models_rotations(
        caplog, partial_had, fused_rotations, add_additional_regions, rotation_mode, equalize_args):
    caplog.set_level(logging.INFO)
    args = equalize_args
    args.rotation_orphan_sink = partial_had
    args.rotation_mode = rotation_mode

    kwargs = {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model = replace_rmsnorm_with_torch(model, model.config)
    model.config.use_cache = False
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=False,
        device=None,
        fuse_sequences=args.fuse_sequences)

    # We need to make sure that the same random matrices are being generated
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # Clone generator to make sure we can use the same rotation matrices
    generator_clone = generator.clone_state()

    # Run model and save outputs
    with torch.no_grad():
        original_logits = model(**calibration_loader[0]).logits

    # Save a copy to apply graph rotation equalization on
    model_copy = copy.deepcopy(model)

    # offload_model is patched to behave as an identity, thus making sure that the operations
    # are deterministic, enabling to test that the tensors match exactly.
    with patch('brevitas_examples.llm.main.offload_model', lambda m: m):
        with patch('brevitas.graph.equalize.random_orthogonal_matrix',
                   partial(_random_orthogonal_matrix, generator=generator)):
            fused_rotation_no_fx(
                model,
                calibration_loader,
                args,
                fuse_rotations=True,
                add_self_attention_regions=add_additional_regions)

    # Run model and save outputs
    with torch.no_grad():
        expected_logits = model(**calibration_loader[0]).logits

    # Instead of random orthogonal matrices, we want to use the same ones as when the activations are not fused.
    with patch('brevitas_examples.llm.main.offload_model', lambda m: m):
        if rotation_mode == 'had':
            with patch('brevitas.graph.equalize._apply_ort_device', _apply_had_device):
                fused_rotation_no_fx(
                    model_copy,
                    calibration_loader,
                    args,
                    fuse_rotations=False,
                    add_self_attention_regions=add_additional_regions)
        else:
            with patch('brevitas.graph.equalize.random_orthogonal_matrix',
                       partial(_random_orthogonal_matrix, generator=generator_clone)):
                fused_rotation_no_fx(
                    model_copy,
                    calibration_loader,
                    args,
                    fuse_rotations=False,
                    add_self_attention_regions=add_additional_regions)

    # Fuse matrices with module weights
    if fused_rotations:
        fuse_rotations(model_copy)

    # Run model and save outputs
    with torch.no_grad():
        logits = model_copy(**calibration_loader[0]).logits

    # Verify that the rotated module output is similar to the original FP
    assert torch.allclose(original_logits, logits, atol=ATOL), "Output of rotated network does not approximately match that of the original network."
    # Verify that the output is the same
    assert torch.allclose(expected_logits, logits, atol=0.0, rtol=0.0), "Outputs of fused/unfused rotated networks do not match exactly."

    num_rotation_matrices = len(extract_trainable_rotation_matrices(model_copy))

    num_rotated_modules = 0
    # Count the number of RotatedModules
    for module in model_copy.modules():
        if isinstance(module, RotatedModule):
            num_rotated_modules += 1

    # Verify that the number of learnable rotation matrices is the expected (R1 + one R2 per block)
    expected_number_rotation_matrices = 0 if fused_rotations else (
        1 + (model.config.num_hidden_layers if add_additional_regions else 0))
    assert num_rotation_matrices == expected_number_rotation_matrices, f"Expected {expected_number_rotation_matrices} learnable rotations, found {num_rotation_matrices}."

    # Verify that the number of rotated modules is correct
    expected_number_rotated_modules = 0 if not partial_had else (
        model.config.num_hidden_layers if add_additional_regions else 2 *
        model.config.num_hidden_layers)
    assert num_rotated_modules == expected_number_rotated_modules, f"Expected {expected_number_rotated_modules} RotatedModules found {num_rotated_modules}."

    # Verify that the weights after fusing match
    for name_fused_module, fused_module in model.named_modules():
        # For linear modules verify that the weights match
        if isinstance(fused_module, (nn.Linear, RotatedModule)):
            for name_unfused_Module, unfused_module in model_copy.named_modules():
                if name_fused_module == name_unfused_Module:
                    # Verify that everything matches between the fused and unfused rotation modules
                    _compare_fused_unfused_rotation_modules(
                        name_fused_module, fused_module, unfused_module)
