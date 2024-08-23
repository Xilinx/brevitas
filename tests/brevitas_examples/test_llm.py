# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from dataclasses import dataclass
import logging
import os
import shutil

import numpy as np
import pytest
import torch

from brevitas import config
from tests.marker import jit_disabled_for_export


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


def allclose(x, y):
    return np.allclose(x, y, rtol=1e-04, atol=3e-00, equal_nan=False)


def allveryclose(x, y):
    return np.allclose(x, y, rtol=1e-08, atol=1e-01, equal_nan=False)


def allexact(x, y):
    return np.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=False)


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


@pytest.fixture(
    scope="session",
    params=[
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            float_ppl=None,
            supports_fx=True,
        ),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-OPTForCausalLM",
            float_ppl=None,
            supports_fx=True,
        ),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-MistralForCausalLM",
            float_ppl=None,
            supports_fx=False,
        ),
        #ModelAndPpl(
        #    name="dacorvo/Mixtral-tiny",
        #    float_ppl=None,
        #    supports_fx=True,
        #),
    ])
def small_models_with_ppl(request):
    yield request.param


@pytest.fixture()
def default_run_args(request):
    from brevitas_examples.llm.main import parse_args
    args = UpdatableNamespace(**vars(parse_args([])))
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
    return args


@pytest.fixture(
    params=[
        {},
        {
            "bias_corr": True},
        {
            "act_equalization": "layerwise"},
        {
            "act_equalization": "fx"},
        {
            "weight_equalization": True},
        {
            "gptq": True},
        {
            "ln_affine_merge": True},])
def toggle_run_args(default_run_args, request):
    args = default_run_args
    args.update(**request.param)
    yield args


@pytest.mark.llm
def test_small_models_toggle_run_args(caplog, toggle_run_args, small_models_with_ppl):
    from brevitas_examples.llm.main import main
    caplog.set_level(logging.INFO)
    args = toggle_run_args
    args.model = small_models_with_ppl.name
    exp_float_ppl = small_models_with_ppl.float_ppl
    use_fx = requires_fx(args)
    if use_fx and not small_models_with_ppl.supports_fx:
        pytest.xfail(f"{small_models_with_ppl.name} does not support FX")
    float_ppl, quant_ppl, model = main(args)


@pytest.fixture(
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_equalization": "layerwise",
            "gptq": True,
            "float_ppl": 31274.05078125,
            "quant_ppl": 33139.23046875},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "fx",
            "bias_corr": True,
            "float_ppl": 33239.5,
            "quant_ppl": 33283.75390625},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",
            "weight_equalization": True,
            "ln_affine_merge": True,
            "replace_mha": True,
            "float_ppl": 50016.0,
            "quant_ppl": 50016.0},])
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
def test_small_models_acc(caplog, acc_args_and_acc):
    from brevitas_examples.llm.main import main
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = acc_args_and_acc
    float_ppl, quant_ppl, model = main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
    if config.JIT_ENABLED:
        assert allclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
        assert allclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"
    else:
        assert allveryclose(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
        assert allveryclose(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"


@pytest.fixture(
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
                    "<class 'brevitas.core.quant.int.RescalingIntQuant'>",}},
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
                    "<class 'brevitas.core.quant.int.RescalingIntQuant'>",}},
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
                    "<class 'brevitas.core.quant.float.FloatQuant'>",}},
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
                    "<class 'brevitas.core.quant.float.FloatQuant'>",}},
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
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.stats_input_view_shape_impl":
                    "<class 'brevitas.core.function_wrapper.shape.OverSubChannelBlockView'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant.scaling_impl.parameter_list_stats.first_tracked_param.view_shape_impl":
                    "<class 'brevitas.core.function_wrapper.shape.OverSubChannelBlockView'>",}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "layerwise",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.equalized_layer.EqualizedModule'>",
                "model.layers.0.self_attn.q_proj.layer":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",}},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",
            "replace_mha": True,
            "exp_layer_types": {
                "model.decoder.layers.0.self_attn":
                    "<class 'brevitas_examples.llm.llm_quant.mha_layers.QuantizableOPTAttention'>",
                "model.decoder.layers.0.self_attn.mha":
                    "<class 'brevitas.nn.quant_mha.QuantMultiheadAttention'>",}},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "quantize_last_layer": True,
            "exp_layer_types": {
                "lm_head": "<class 'brevitas.nn.quant_linear.QuantLinear'>"}},])
def layer_args(default_run_args, request):
    args = default_run_args
    layer_dict = request.param
    exp_layer_types = layer_dict["exp_layer_types"]
    del layer_dict["exp_layer_types"]
    args.update(**layer_dict)
    yield args, exp_layer_types


@pytest.mark.llm
def test_small_models_quant_layer(caplog, layer_args):
    from brevitas_examples.llm.main import main
    caplog.set_level(logging.INFO)
    args, exp_layer_types = layer_args
    float_ppl, quant_ppl, model = main(args)
    assert_layer_types(model, exp_layer_types)


@pytest.fixture(
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
def test_small_models_onnx_export(caplog, onnx_export_args):
    from brevitas_examples.llm.main import main
    import onnx
    caplog.set_level(logging.INFO)
    args = onnx_export_args
    float_ppl, quant_ppl, model = main(args)
    onnx_model = onnx.load(os.path.join(args.export_prefix, "model.onnx"))
    shutil.rmtree(args.export_prefix)


@pytest.fixture(
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
def test_small_models_torch_export(caplog, torch_export_args):
    from brevitas_examples.llm.main import main
    caplog.set_level(logging.INFO)
    args = torch_export_args
    float_ppl, quant_ppl, model = main(args)
    filepath = args.export_prefix + ".pt"
    torchscript_model = torch.jit.load(filepath)
    os.remove(filepath)
