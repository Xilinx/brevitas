# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest_cases

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
    def case_small_models_with_ppl(self, run_dict, default_run_args, request):
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
                "learned_round": "linear_round",
                "learned_round_iters": 1,
                "gpxq_block_name": "model.layers",
            },
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
            "act_equalization=fx,gptq=True",
            "quant_sdpa_fx_per_row",
            "quant_sdpa_functional_per_row",
            "functional_sdpa_quant=True,rotation=fused_no_fx",
            "per_group_w_padding,learned_round=linear_round",
        ],)
    def case_small_models_toggle_args(self, run_dict, default_run_args, request):
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
                "quant_sdpa": "fx",
                "float_ppl": 51649.797,
                "quant_ppl": 51688.922},
        ],
        ids=[
        "llama",
        "llama_float_dynamic_input",
        "mistral",
        "opt-quant-sdpa",],)
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
                "quant_ppl": 36821.664}
        ],
        ids=[
        "llama",
        "mistral",],)
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
            "quant_ppl": 32515.525,},
        ],
        ids=[
        "llama_fused_rotation_ort",
        "llama_fused_rotation_ort_no_orphan",
        "llama_fused_rotation_had",
        "llama_fused_rotation_had_no_orphan",
        "llama_layerwise",
        "llama_fused_rotation_had_no_orphan_expanded"],)
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
        ],
        ids=[
            "mistral-int8",
            "mistral-weight-only",
            "mistral-fp8_ocp",
            "mistral-fp8_fnuz",
            "llama-mxfp8",
            "llama-int8-act_equalization=layerwise",
            "mistral-int8-quant-last-layer",
            "llama-int8-svd_quant",
            "opt-quant-sdpa",],)
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
            "float_ppl": 32428.475,
            "quant_ppl": 32357.392578125,
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
            "float_ppl": 32428.475,
            "quant_ppl": 32452.111,
            "exp_layer_types_count": {
                "<class 'brevitas.nn.equalized_layer.RotatedModule'>": 0,
                "<class 'torch.nn.utils.parametrize.ParametrizedLinear'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedEmbedding'>": 1,
                "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>": 14,}},],
        ids=[
        "llama_rotation_optimization_ort",
        "llama_rotation_optimization_ort_no_orphan",
        "llama_rotation_optimization_had",
        "llama_rotation_optimization_had_sdpa",
        "llama_rotation_optimization_had_no_orphan",],)
    def case_small_models_rotation_optimization(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=LLMPerplexityCases.METRICS+["exp_layer_types_count"])
