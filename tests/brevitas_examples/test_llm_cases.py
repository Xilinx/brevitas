# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases

from brevitas import config
from tests.brevitas_examples.common import process_args_and_metrics


class LLMRunCases:

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",},
            {
                "model": "hf-internal-testing/tiny-random-MistralForCausalLM",},
            # Ready for MoE support
            #{
            #    "model": ""dacorvo/Mixtral-tiny",},
            {
                "model": "hf-internal-testing/tiny-random-OPTForCausalLM",},],
        ids=[
            "llama",
            "mistral",  #"mixtral",
            "opt",],
    )
    def case_small_models_run(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict)

    # yapf: disable
    @pytest_cases.parametrize(
        "run_dict",
        [
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
            {"rotation": "layerwise", "act_equalization": "layerwise", "convert_layernorm_to_rmsnorm": True},
            {"act_equalization": "fx", "gptq": True},
            {"quant_sdpa": "fx", "input_scale_type": "dynamic", "input_quant_granularity": "per_row"},
            {"quant_sdpa": "functional", "input_scale_type": "dynamic", "input_quant_granularity": "per_row"},
            {
                "quant_sdpa": "functional",
                "rotation": "fused_no_fx",
                "rotation_sdpa_regions": True,
                "input_scale_type": "dynamic",
                "replace_rmsnorm": True
            }, {
                "weight_quant_granularity": "per_group",
                "weight_group_size": 11,
                "learned_round": "identity",
                "learned_round_iters": 1,
                "gpxq_block_name": "model.layers",
            },{
                "weight_quant_format": "float_e2m1",
                "weight_param_method": "mse",
            },
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "weight_bit_width": 3,
                "weight_group_size": 128,
                "weight_quant_granularity": "per_group",
                "weight_quant_type": "asym",
                "scaling_min_val": 1e-5,
                "quantize_weight_zero_point": True,
                "awq_scale": True,
                "awq_clip": True,
            }
        ],
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
            "rotation=layerwise,act_equalization=layerwise,convert_layernorm_to_rmsnorm=True",
            "act_equalization=fx,gptq=True",
            "quant_sdpa_fx_per_row",
            "quant_sdpa_functional_per_row",
            "functional_sdpa_quant=True,rotation=fused_no_fx",
            "per_group_w_padding,learned_round=identity",
            "float_e2m1_and_mse",
            "awq_clip_scale"
        ],)
    def case_small_models_toggle_args(self, run_dict, default_run_args, request):
        if config.JIT_ENABLED and run_dict.get("weight_param_method") == "mse":
            pytest.skip(reason=f'MSE as weight_param_method requires JIT to be disabled')
        yield process_args_and_metrics(default_run_args, run_dict)

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "custom_quantizer": "example_int8_weight_quant",},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "custom_quantizer": "tests/brevitas_examples/llm_test_plugin.py:example_int4_weight_quant"},],
        ids=[
            "llama-quant", "llama-quant-file",]
    )
    def case_small_models_custom_quantizer(self, run_dict, default_run_args, request):
        from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
        from brevitas.utils.python_utils import Registry
        from brevitas_examples.common.generative.quantizers import BaseQuantizer
        from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
        @Registry.register(QUANTIZERS_REGISTRY, "example_int8_weight_quant")
        class ExampleInt8WeightQuantizer(BaseQuantizer):
            weight_quant = Int8WeightPerTensorFloat
        yield process_args_and_metrics(default_run_args, run_dict)


class LLMPerplexityCases:

    METRICS = ["float_ppl", "quant_ppl"]

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_equalization": "fx",
                "bias_corr": True,
                "float_ppl": 30795.76953125,
                "quant_ppl": 30861.037109375},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_equalization": "fx",
                "bias_corr": True,
                "weight_quant_format": "float_ocp_e4m3",
                "input_quant_format": "float_ocp_e4m3",
                "input_quant_granularity": "per_row",
                "input_scale_type": "dynamic",
                "input_quant_type": "sym",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30793.537109375},
            {
                "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
                "act_equalization": "layerwise",
                "gptq": True,
                "float_ppl": 30977.689453125,
                "quant_ppl": 30958.1953125},
            {
                "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
                "weight_equalization": True,
                "ln_affine_merge": True,
                "quant_sdpa": "fx",
                "float_ppl": 46088.265625,
                "quant_ppl": 46327.50390625},
            {
                "model": "hf-internal-testing/tiny-random-OPTForCausalLM",  # Requires PT>=2.4 to run
                "calibration_batch_size": 2,
                "seqlen": 4,
                "gptq": True,
                "ln_affine_merge": True,
                "convert_layernorm_to_rmsnorm": True,
                "replace_rmsnorm": True,
                "rotation": "fx",
                "float_ppl": 54132.29296875,
                "quant_ppl": 54140.08984375},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "weight_bit_width": 2,
                "weight_scale_precision": "signed_float_scale",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30970.068359375},
        ],
        ids=[
        "llama",
        "llama_float_dynamic_input",
        "mistral",
        "opt-quant-sdpa",
        "rotation_fx_and_gptq",
        "llama_signed_scale",
        ],)
    def case_small_models_with_ppl(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=LLMPerplexityCases.METRICS)

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "learned_round": "identity",
                "learned_round_iters": 1,
                "gpxq_block_name": "model.layers",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30675.064453125},
            {
                "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "learned_round": "identity",
                "learned_round_iters": 1,
                "gpxq_block_name": "model.layers",
                "float_ppl": 30977.689453125,
                "quant_ppl": 30952.52734375}
        ],
        ids=[
        "llama",
        "mistral",
        ],)
    def case_small_models_learned_round_ppl(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=LLMPerplexityCases.METRICS)

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "replace_rmsnorm": True,
                "rotation": "fused_no_fx",
                "rotation_orphan_sink": True,
                "rotation_mode": "ort",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30991.04296875,},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "replace_rmsnorm": True,
                "rotation": "fused_no_fx",
                "rotation_orphan_sink": False,
                "rotation_mode": "ort",
                "float_ppl": 30795.76953125,
                "quant_ppl": 31010.615234375,},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "replace_rmsnorm": True,
                "rotation": "fused_no_fx",
                "rotation_orphan_sink": True,
                "rotation_mode": "had",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30956.54296875,},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "replace_rmsnorm": True,
                "rotation": "fused_no_fx",
                "rotation_orphan_sink": False,
                "rotation_mode": "had",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30836.9140625},
            {
                "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "act_calibration": False,
                "weight_bit_width": 4,
                "input_bit_width": None,
                "replace_rmsnorm": True,
                "rotation": "layerwise",
                "float_ppl": 30795.76953125,
                "quant_ppl": 30829.4453125,},
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
                "float_ppl": 30795.76953125,
                "quant_ppl": 30830.03125,},
        ],
        ids=[
        "llama_fused_rotation_ort",
        "llama_fused_rotation_ort_no_orphan",
        "llama_fused_rotation_had",
        "llama_fused_rotation_had_no_orphan",
        "llama_layerwise",
        "llama_fused_rotation_had_no_orphan_expanded"
        ],)
    def case_small_models_rotation_ppl(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=LLMPerplexityCases.METRICS)

class LLMQuantLayerTypeCases:

    @pytest_cases.parametrize(
        "run_dict",
        [
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
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "rotation": "layerwise",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.equalized_layer.RotatedModule'>",
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
            "quant_sdpa": "fx",
            "exp_layer_types": {
                "attn_output": "<class 'brevitas.nn.quant_sdpa.QuantScaledDotProductAttention'>",}},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_scale_precision": "po2_scale",
            "weight_param_method": "stats",
            "weight_quant_granularity": "per_group",
            "weight_group_size": 16,
            "weight_quant_type": "sym",
            "weight_param_method": "mse",
            "input_quant_format": "float_ocp_e5m2",
            "input_scale_type": "dynamic",
            "input_scale_precision": "po2_scale",
            "input_param_method": "stats",
            "input_quant_granularity": "per_group",
            "input_group_size": 16,
            "input_quant_type": "sym",
            "act_calibration": False,
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl":
                    "<class 'brevitas.core.stats.stats_op.MSE'>",},},
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "weight_quant_format": "float_ocp_e4m3",
            "weight_quant_type": "sym",
            "weight_scale_precision": "signed_float_scale",
            "input_quant_format": "float_ocp_e5m2",
            "input_quant_type": "sym",
            "input_scale_precision": "signed_float_scale",
            "exp_layer_types": {
                "model.layers.0.self_attn.q_proj":
                    "<class 'brevitas.nn.quant_linear.QuantLinear'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.stats.stats_impl":
                    "<class 'brevitas.core.stats.stats_op.SignedAbsMax'>",
                "model.layers.0.self_attn.q_proj.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.restrict_scaling.restrict_value_impl":
                    "<class 'brevitas.core.restrict_val.SignedFloatRestrictValue'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant":
                    "<class 'brevitas.core.quant.float.FloatQuant'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl":
                    "<class 'brevitas.core.stats.stats_op.SignedAbsMax'>",
                "model.layers.0.self_attn.q_proj.weight_quant.tensor_quant.scaling_impl.stats_scaling_impl.restrict_clamp_scaling.restrict_value_impl":
                    "<class 'brevitas.core.restrict_val.SignedFloatRestrictValue'>",
                },
            },
        ],
        ids=[
            "mistral-int8",
            "mistral-weight-only",
            "mistral-fp8_ocp",
            "mistral-fp8_fnuz",
            "llama-mxfp8",
            "llama-int8-act_equalization=layerwise",
            "llama-int8-rotation=layerwise",
            "mistral-int8-quant-last-layer",
            "llama-int8-svd_quant",
            "opt-quant-sdpa",
            "llama-mxfp8-mse",
            "mistral-fp8_ocp-signed",
        ],)
    def case_small_models_quant_layer(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=["exp_layer_types"])

class LLMQuantLayerCountCases:

    @pytest_cases.parametrize(
        "run_dict",
        [
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
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 2,  # Sinks: Down proj
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
                "<class 'brevitas.nn.quant_linear.QuantLinear'>": 14,}},
        ],
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
        "llama-int8-svd_quant"],)
    def case_small_models_quant_layer_types_count(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=["exp_layer_types_count"])


class LLMRotationOptimizationCases:

    @pytest_cases.parametrize(
        "run_dict",
            [
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30973.669921875,
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30941.7265625,
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30656.814453125,
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
                    "rotation_sdpa_regions": True,
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30851.2089843750,
                    "exp_layer_types_count": {
                        "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 2,
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
                    "rotation_sdpa_regions": True,
                    "optimize_rotations": True,
                    "rotation_orphan_sink": True,
                    "rotation_mode": "had",
                    "rotation_block_size": 32,
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30850.916015625,
                    "exp_layer_types_count": {
                        "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 2,
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
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30751.923828125,
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
                    "rotation_orphan_sink": False,
                    "rotation_mode": "had",
                    "nsamples_rot_calibration": 2,
                    "dtype": "float32",
                    "extra_args": [
                        "--learning_rate",
                        "1.5",
                        "--gamma",
                        "0.0",
                        "--use-distillation-loss",
                        "True",
                        "--max_steps",
                        "2",
                        "--per_device_train_batch_size",
                        "1",
                        "--gradient_accumulation_steps",
                        "1"],
                    "float_ppl": 30795.76953125,
                    "quant_ppl": 30688.232421875,
                    "exp_layer_types_count": {
                        "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 0,
                        "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                        "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                        "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},
        ],
        ids=[
        "llama_rotation_optimization_ort",
        "llama_rotation_optimization_ort_no_orphan",
        "llama_rotation_optimization_had",
        "llama_rotation_optimization_had_sdpa",
        "llama_rotation_optimization_had_sdpa_blockwise",
        "llama_rotation_optimization_had_no_orphan",
        "llama_rotation_optimization_had_no_orphan_distillation_loss"],)
    def case_small_models_rotation_optimization(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=LLMPerplexityCases.METRICS+["exp_layer_types_count"])
