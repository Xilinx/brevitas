# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from copy import deepcopy
import functools
import sys
from warnings import warn

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import numpy as np
from optimum.exporters.onnx import onnx_export_from_model
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils.fx import _SUPPORTED_MODELS
import yaml

from brevitas.export import export_torch_qcdq
from brevitas.export.inference.manager import quant_inference_mode
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import LayerwiseActivationRotation
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.utils import get_module
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import compute_perplexity
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.gpxq import apply_gpfq
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.learned_round_utils import apply_learned_round
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.prepare_for_quantize import \
    replace_sdpa_with_quantizable_layers
from brevitas_examples.llm.llm_quant.run_utils import CastFloat16ToFloat32
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.llm_quant.run_utils import get_fx


def filter_results(results, tasks):
    # filter out what we actually want to track in azureml
    eval_results = dict()
    for task_name in tasks:
        # first, log n_shots for each task
        # for subtask, n_shots in results["n-shot"].items():
        #     name = f"{subtask}_n_shot"
        #     eval_results[name] = float(n_shots)
        # then log all result metrics we have for this task
        for key, val in results["results"][task_name].items():
            if not isinstance(val, str):
                # for mmlu, we don't log results per subtask, but simply overall results
                name = f"{task_name}_{key}"
                eval_results[name] = val
    return eval_results


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def fused_rotation_no_fx(model, calibration_loader, args):
    with torch.no_grad():
        new_model, guards = torch._dynamo.export(model)(**calibration_loader[0])
    apply_layernorm_affine_merge(new_model)
    new_model, rewriters = apply_layernorm_to_rmsnorm(new_model, return_rewriters=True)
    rewriters = fix_rewriter(rewriters, model, 'weight')

    for r in rewriters:
        r.apply(model)
    new_model = offload_model(new_model)
    eq = GraphRotationEqualization(
        orphan_sink=args.rotation_orphan_sink,
        full_rotation_method=args.rotation_mode,
        return_rewriters=True,
        sdpa_regions=args.rotation_sdpa_regions)
    new_model, rewriters = eq.apply(new_model)
    rewriters = fix_rewriter(rewriters, model, 'weight')

    for r in rewriters:
        r.apply(model)
    remove_hooks(new_model)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def model_export(model, ref_input, args):
    if args.export_target == 'sharded_torchmlir_group_weight':
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import \
            sharded_weight_group_export
        sharded_weight_group_export(model, no_custom_packed_export=True)
    elif args.export_target == 'sharded_packed_torchmlir_group_weight':
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import \
            sharded_weight_group_export
        sharded_weight_group_export(model, no_custom_packed_export=False)
    elif args.export_target == 'onnx_qcdq':
        if args.weight_quant_granularity == 'per_group':
            export_manager = BlockQuantProxyLevelManager
        else:
            export_manager = StdQCDQONNXManager
            export_manager.change_weight_export(export_weight_q_node=True)

        print(f"Exporting the model in ./{args.export_prefix}")
        with torch.no_grad(), brevitas_proxy_export_mode(model, export_manager=export_manager):
            onnx_export_from_model(
                model,
                f"./{args.export_prefix}",
                task="text-generation-with-past",
                do_validation=False)
    elif args.export_target == 'torch_qcdq':
        export_torch_qcdq(model, ref_input['input_ids'], export_path=f"{args.export_prefix}.pt")


def validate(args):
    if args.rotation == 'fx':
        assert args.ln_affine_merge, 'Graph rotation requires to merge LN/RMS norm affine parameters'
        assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
        assert args.convert_layernorm_to_rmsnorm, 'Graph rotation requires to replace LayerNorm with RMSNorm'
    elif args.rotation == 'fused_no_fx':
        assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
    if not args.no_quantize:
        if args.gptq and args.gpfq:
            warn("Both GPTQ and GPFQ are enabled.")
        if args.gpxq_max_accumulator_bit_width is not None:
            assert args.weight_quant_format == 'int', "AXE only supports integer formats."
            assert args.input_quant_format == 'int', "AXE only supports integer formats."
            assert args.input_bit_width is not None, \
                "Specify input bit width; activation quantization is required to guarantee accumulator bounds."
            if not (args.gptq or args.gpfq):
                warn("Max accumulator bit width is specified, but no GPxQ is enabled.")
            if args.gpxq_max_accumulator_tile_size is not None:
                if args.weight_quant_granularity == 'per_group':
                    assert args.gpxq_max_accumulator_tile_size == args.weight_group_size, \
                        "Group size must be equal to tile size with per_group quantization."
                if args.input_quant_granularity == 'per_group':
                    assert args.gpxq_max_accumulator_tile_size == args.input_group_size, \
                        "Group size must be equal to tile size with per_group quantization."
        if args.export_target is not None:
            assert args.input_quant_format == 'int', "Only integer quantization supported for export currently."
        if args.export_target is not None and args.input_bit_width is not None:
            assert args.input_scale_type == 'static', "Only static scale supported for export currently."
        if args.export_target == 'sharded_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'sharded_packed_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded packed torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'onnx_qcdq':
            if args.weight_quant_granularity == 'per_group':
                assert args.input_bit_width is None, "ONNX QCDQ per_group quantization requires no input quantization"
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_bit_width is not None and args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.export_target == 'torch_qcdq':
            assert args.weight_quant_granularity != 'per_group', "TorchScript QCDQ export doesn't support group weight quantization."
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_bit_width is not None and args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.input_bit_width and args.input_scale_type == 'static':
            assert args.act_calibration, "Static input quantization is being applied without activation calibration. Set --act-calibration."
        if (args.weight_equalization or args.act_equalization == 'fx'):
            if args.replace_mha:
                assert args.export_target != 'onnx_qcdq', "Cannot export ONNX QCDQ with FX + MHA replacing"
            else:
                assert args.export_target != 'torch_qcdq', "Cannot export Torch QCDQ with FX"

    if not args.fuse_sequences:
        # 350 is approximately the 99% percentile for the sequence length in WikiText2 (train partition, using AutoTokenizer)
        if args.seqlen >= 350:
            warn(
                "Data loading can take a long time or, potentially, enter an infinite loop. Consider setting --args.fuse_sequences "
                "or decreasing the sequence length (seqlen)")


def quantize_llm(args):
    validate(args)
    set_seed(args.seed)
    if args.export_prefix is None:
        args.export_prefix = f"{args.model.replace('/', '--')}"

    if args.no_float16:
        dtype = torch.float32
    else:
        dtype = torch.float16

    # Whether to quantize SDPA with FX
    quant_sdpa_fx = args.quant_sdpa and not args.replace_mha

    kwargs = {"torch_dtype": dtype}
    if quant_sdpa_fx:
        kwargs["attn_implementation"] = "sdpa"

    if args.export_target == 'torch_qcdq':
        kwargs['torchscript'] = True

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    float_ppl = None
    quant_ppl = None

    if args.load_awq:
        from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
        awq_results = torch.load(args.load_awq, map_location="cpu")
        with CastFloat16ToFloat32():
            apply_awq(model, awq_results)

    require_fx = True if args.weight_equalization or args.act_equalization == 'fx' or args.ln_affine_merge or args.convert_layernorm_to_rmsnorm or quant_sdpa_fx else False

    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=require_fx and args.export_target is not None,
        device=None,
        fuse_sequences=args.fuse_sequences)

    validation_loader = get_dataset_for_model(
        args.model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="validation",
        seed=args.seed,
        require_fx=require_fx and args.export_target is not None,
        device=None,
        fuse_sequences=args.fuse_sequences)

    device = next(iter(model.parameters())).device
    print("Data loaded.")

    if args.eval:
        assert args.export_target != 'torch_qcdq', "TorchScript QCDQ export and Evaluation simultaneously"
        print("Float model eval...")
        model = offload_model(model)
        float_ppl = compute_perplexity(
            model, validation_loader, context_length=args.seqlen // 2, tokenizer=tokenizer)
        remove_hooks(model)
        print(f"Float perplexity ({args.dataset}): {float_ppl:.3f}")

    if args.replace_rmsnorm:
        model = replace_rmsnorm_with_torch(model, model.config)

    if require_fx:
        if model.__class__.__name__ in _SUPPORTED_MODELS and not args.replace_rmsnorm:
            model = get_fx(model, is_export=args.export_target is not None)
        else:
            with torch.no_grad():
                model, guards = torch._dynamo.export(model)(**calibration_loader[0])
        # Blockwise optimization does not work with FX at the moment
        args.gpxq_block_name = None

    # Apply LN affine merging before inserting MHA layers
    # since currently there is support only for merging into Linear
    if args.ln_affine_merge:
        print("Apply LN affine merge...")
        apply_layernorm_affine_merge(model)
        print("LN affine merge applied.")

    if args.convert_layernorm_to_rmsnorm:
        print("Convert LayerNorm to RMSNorm...")
        apply_layernorm_to_rmsnorm(model)
        print("Layernorm To RMSNorm applied.")

    if args.rotation == 'fx':
        model = offload_model(model)
        eq = GraphRotationEqualization(
            orphan_sink=args.rotation_orphan_sink,
            full_rotation_method=args.rotation_mode,
            sdpa_regions=args.rotation_sdpa_regions)
        model = eq.apply(model)
        remove_hooks(model)
    elif args.rotation == 'layerwise':
        eq = LayerwiseActivationRotation()
        model = eq.apply(model)
    elif args.rotation == 'fused_no_fx':
        fused_rotation_no_fx(model, calibration_loader, args)

    # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
    # with all the variability in HF implementations
    if args.replace_mha:
        print("Replace HF MHA with quantizable variants...")
        model = replace_mha_with_quantizable_layers(model, dtype)
        print("Replacing done.")
    elif quant_sdpa_fx:
        print("Replace `F.scaled_dot_product_attention` with QuantSDPA...")
        model = replace_sdpa_with_quantizable_layers(model)
        print("Replacing done.")

    if args.weight_equalization:
        print("Apply weight equalization...")
        # In case of float16 model, we need to offload to account for missing ops
        model = offload_model(model)
        apply_weight_equalization(model)
        remove_hooks(model)
        print("Weight equalization applied.")

    if args.act_equalization is not None:
        offload_model(model)
        print(f"Apply act equalization (SmoothQuant) with alpha {args.act_equalization_alpha}")
        apply_act_equalization(
            model, args.act_equalization, calibration_loader, alpha=args.act_equalization_alpha)
        print("Act equalization applied.")
        remove_hooks(model)

    if not args.no_quantize:
        name_blacklist = []
        print("Applying model quantization...")
        linear_input_quant, weight_quant, input_quant, q_scaled_quant, k_transposed_quant, v_quant, attn_output_weights_quant = generate_quantizers(
            dtype=dtype,
            weight_bit_width=args.weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_type=args.weight_quant_type,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            weight_group_dim=args.weight_group_dim,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            weight_quant_format=args.weight_quant_format,
            input_bit_width=args.input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_precision=args.input_scale_precision,
            input_scale_type=args.input_scale_type,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            input_group_size=args.input_group_size,
            quantize_input_zero_point=args.quantize_input_zero_point,
            scale_rounding_func_type=args.scale_rounding_func_type,
            device=device,
            scaling_min_val=args.scaling_min_val)
        layer_map = generate_quant_maps(
            linear_input_quant=linear_input_quant,
            weight_quant=weight_quant,
            input_quant=input_quant,
            q_scaled_quant=q_scaled_quant,
            k_transposed_quant=k_transposed_quant,
            v_quant=v_quant,
            attn_output_weights_quant=attn_output_weights_quant,
            dtype=dtype,
            device=device,
            input_quant_format=args.input_quant_format,
            quantize_embedding=False)
        if not args.quantize_last_layer:
            if require_fx:
                last_node = [node for node in model.graph.nodes if node.op == 'call_module'][-1]
                last_module = get_module(model, last_node.target)
                last_layer_kwargs = layer_map[type(last_module)][1]
                prev_weight_quant = deepcopy(last_layer_kwargs['weight_quant'])
                prev_input_quant = deepcopy(last_layer_kwargs['input_quant'])
                weight_quant = lambda module: prev_weight_quant if id(module) != id(
                    last_module) else None
                input_quant = lambda module: prev_input_quant if id(module) != id(
                    last_module) else None
                last_layer_kwargs['weight_quant'] = weight_quant
                last_layer_kwargs['input_quant'] = input_quant
            else:
                name_blacklist += ["lm_head", "embed_out"]
        model = layerwise_quantize(
            model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
        # Tie back first/last layer weights in case they got untied
        print("Model quantization applied.")

    # If any equalization has taken places, the embedding layer and the fully connected one are
    # not tied anymore, and they need to be treated as standalone, separate layers.
    # In all other cases we can tie them back so to preserve memory.
    if args.act_equalization is None and not require_fx and args.rotation is None:
        model.tie_weights()

    if args.bias_corr:
        model = add_zero_bias_to_linear(model)

    model = offload_model(model)

    dict_hooks = dict()

    # When offloading to CPU + GPU, the CPU scale factors must be updated
    # before we move them back to the meta device.
    # If we don't, we lose the new value but the internal flag "init_done" is True, thus we will use the wrong scale.
    # To do this, we attach a "hook" to the post_forward function, called before the post_forward
    # The function will update the dict with the initialized scales
    for m in model.modules():
        if hasattr(m, '_hf_hook'):
            if m._hf_hook.weights_map is not None:
                # We store the original function to be restored later
                dict_hooks[m] = m._hf_hook.post_forward
                new_funct = functools.partial(update_internal_dict, m)
                m._hf_hook.post_forward = hooked_on_a_function(m._hf_hook.post_forward, new_funct)

    with torch.no_grad():
        model(**calibration_loader[0])

    # We restore the original behaviour of the post-forward.
    for k, v in dict_hooks.items():
        k._hf_hook.post_forward = v

    if args.act_calibration:
        print("Apply act calibration...")
        apply_calibration(model, calibration_loader)
        print("Act calibration applied.")

    if args.learned_round:
        print("Applying learned round...")
        remove_hooks(model)
        apply_learned_round(
            model,
            calibration_loader,
            iters=args.learned_round_iters,
            block_name_attribute=args.gpxq_block_name,
            learn_scale=args.learned_round_scale,
            scale_optimizer_class='sgd',
            optimizer_kwargs={'lr': args.learned_round_lr},
            scale_optimizer_kwargs={
                'lr': args.learned_round_scale_lr, 'momentum': args.learned_round_scale_momentum},
            fast_update=args.learned_round_fast_update)
        print("Learned round applied.")

        model = offload_model(model)

    if args.gptq:
        print("Applying GPTQ...")
        apply_gptq(
            model,
            calibration_loader,
            act_order=args.gpxq_act_order,
            use_quant_activations=args.gpxq_use_quant_activations,
            create_weight_orig=args.gpxq_create_weight_orig,
            block_name=args.gpxq_block_name,
            max_accumulator_bit_width=args.gpxq_max_accumulator_bit_width,
            max_accumulator_tile_size=args.gpxq_max_accumulator_tile_size)
        print("GPTQ applied.")

    if args.gpfq:
        print("Applying GPFQ...")
        apply_gpfq(
            model,
            calibration_loader,
            act_order=args.gpxq_act_order,
            block_name=args.gpxq_block_name,
            max_accumulator_bit_width=args.gpxq_max_accumulator_bit_width,
            max_accumulator_tile_size=args.gpxq_max_accumulator_tile_size)
        print("GPFQ applied.")

    if args.bias_corr:
        print("Applying bias correction...")
        apply_bias_correction(model, calibration_loader)
        print("Bias correction applied.")

    if args.eval and not args.no_quantize:
        print("Model eval...")
        with torch.no_grad(), quant_inference_mode(model):
            model(**calibration_loader[0])
            quant_ppl = compute_perplexity(
                model, validation_loader, context_length=args.seqlen // 2, tokenizer=tokenizer)
        print(f"Quantized perplexity ({args.dataset}): {quant_ppl:.3f}")

    if args.few_shot_eval:
        with torch.no_grad(), quant_inference_mode(model):
            model(**calibration_loader[0])
            if args.few_shot_compile:
                remove_hooks(model)
                model.cuda()
                model = torch.compile(model)

            wrapped_model = HFLM(pretrained=model)  # need to wrap for LLM eval
            results = evaluator.simple_evaluate(
                model=wrapped_model,
                model_args=None,
                tasks=list(args.few_shot_tasks),
                device='cuda:0',
                limit=args.few_shot_limit,
                num_fewshot=0 if args.few_shot_zeroshot else None,
                log_samples=False,
                batch_size=None,
                verbosity="ERROR")
        results = filter_results(results, args.few_shot_tasks)
        print("Few shot eval results")
        print(results)
    remove_hooks(model)

    if args.checkpoint_name is not None:
        print(f"Saving checkpoint to {args.checkpoint_name}")
        torch.save(model.state_dict(), args.checkpoint_name)

    if args.export_target:
        print(f"Export to {args.export_target}")
        # Currently we always export on CPU with a float32 container to avoid float16 CPU errors
        model = model.to(dtype=torch.float32)
        model_export(model, calibration_loader[0], args)

    return float_ppl, quant_ppl, model


def override_defaults(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify alternative default commandline args (e.g., config/default_template.yml). Default: %(default)s.'
    )
    known_args = parser.parse_known_args()[0]  # Returns a tuple
    if known_args.config is not None:
        with open(known_args.config, 'r') as f:
            defaults = yaml.safe_load(f)
    else:
        defaults = {}
    return defaults


def parse_args(args, override_defaults={}):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify alternative default commandline args (e.g., config/default_template.yml). Default: %(default)s.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="facebook/opt-125m",
        help='HF model name. Default: facebook/opt-125m.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data. Default: 0.')
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help='Number of calibration data samples. Default: 128.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length. Default: 2048.')
    parser.add_argument('--eval', action='store_true', help='Eval model PPL on the chosen Dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wikitext2', 'c4'],
        default='wikitext2',
        help='Dataset to use for quantization (default: %(default)s)')
    parser.add_argument(
        '--gpxq-block-name',
        type=str,
        default=None,
        help=
        'Block name for faster GPxQ optimization. It works only if FX is not needed (default: %(default)s)'
    )
    parser.add_argument(
        '--weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
    parser.add_argument(
        '--weight-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse', 'hqo'],
        help='How scales/zero-point are determined. Default: stats.')
    parser.add_argument(
        '--weight-scale-precision',
        type=str,
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help='Whether scale is a float value or a po2. Default: po2.')
    parser.add_argument(
        '--weight-quant-type',
        type=str,
        default='sym',
        choices=['sym', 'asym'],
        help='Weight quantization type. Default: asym.')
    parser.add_argument(
        '--weight-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
    parser.add_argument(
        '--weight-quant-granularity',
        type=str,
        default='per_group',
        choices=['per_channel', 'per_tensor', 'per_group'],
        help='Granularity for scales/zero-point of weights. Default: per_group.')
    parser.add_argument(
        '--scale-rounding-func-type',
        type=str,
        default=None,
        choices=['round', 'ceil', 'floor'],
        help='Rounding function to use with Po2 scale. Default: None.')
    parser.add_argument(
        '--weight-group-dim',
        type=int,
        default=None,
        choices=[1, 0],
        help='Override default group_dim for groupsize quantization. Default: layer-dependant')
    parser.add_argument(
        '--weight-group-size',
        type=int,
        default=128,
        help='Group size for per_group weight quantization. Default: 128.')
    parser.add_argument(
        '--quantize-weight-zero-point', action='store_true', help='Quantize weight zero-point.')
    parser.add_argument(
        '--input-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (disables input quantization).')
    parser.add_argument(
        '--input-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Input quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
    parser.add_argument(
        '--input-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help=
        'How scales/zero-point are determined. Default: stats (percentile for static, absmax or minmax for dynamic).'
    )
    parser.add_argument(
        '--input-scale-precision',
        type=str,
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help='Whether input scale is a float value or a po2. Default: float.')
    parser.add_argument(
        '--input-scale-type',
        type=str,
        default='static',
        choices=['static', 'dynamic', 'no_scale'],
        help='Whether input scale is a static value or a dynamic value.')
    parser.add_argument(
        '--input-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Input quantization type. Default: asym.')
    parser.add_argument(
        '--input-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor', 'per_row', 'per_group'],
        help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
    parser.add_argument(
        '--input-group-size',
        type=int,
        default=64,
        help='Group size for per_group input quantization. Default: 64.')
    parser.add_argument(
        '--learned-round-lr',
        type=float,
        default=5e-3,
        help='Learning rate for learned round parameter optimization. Default: %(default)s')
    parser.add_argument(
        '--learned-round-scale-lr',
        type=float,
        default=1e-2,
        help='Learning rate for scale optimization during round learning. Default: %(default)s')
    parser.add_argument(
        '--learned-round-scale-momentum',
        type=float,
        default=0.9,
        help='Learning rate for scale optimization during round learning. Default: %(default)s')
    parser.add_argument(
        '--learned-round-iters',
        type=int,
        default=200,
        help='Number of iterations for learned round. Default: 200.')
    parser.add_argument(
        '--learned-round-scale',
        action='store_true',
        help='Learned scale factor together with round.')
    parser.add_argument(
        '--quantize-input-zero-point', action='store_true', help='Quantize input zero-point.')
    parser.add_argument(
        '--quantize-last-layer', action='store_true', help='Quantize last nn.Linear layer.')
    parser.add_argument('--gptq', action='store_true', help='Apply GPTQ.')
    parser.add_argument('--gpfq', action='store_true', help='Apply GPFQ.')
    parser.add_argument(
        '--gpxq-act-order', action='store_true', help='Apply GPxQ activation ordering.')
    parser.add_argument(
        '--gpxq-use-quant-activations',
        action='store_true',
        help='Use quantized activations in GPxQ.')
    parser.add_argument(
        '--gpxq-create-weight-orig', action='store_true', help='Create weight_orig in GPxQ.')
    parser.add_argument(
        '--gpxq-max-accumulator-bit-width',
        type=int,
        default=None,
        help='Maximum accumulator bit width for GPxQ using AXE.')
    parser.add_argument(
        '--gpxq-max-accumulator-tile-size',
        type=int,
        default=None,
        help='Maximum accumulator tile size for GPxQ using AXE.')
    parser.add_argument(
        '--act-calibration', action='store_true', help='Apply activation calibration.')
    parser.add_argument('--bias-corr', action='store_true', help='Apply bias correction.')
    parser.add_argument('--ln-affine-merge', action='store_true', help='Merge LN affine params.')
    parser.add_argument(
        '--convert-layernorm-to-rmsnorm', action='store_true', help='Merge LN affine params.')
    parser.add_argument(
        '--replace-rmsnorm', action='store_true', help='Replace HF RMSNorms with Torch one.')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization.')
    parser.add_argument(
        '--no-float16',
        action='store_true',
        help='Disable float16 as base datatype and switch to float32.')
    parser.add_argument(
        '--scaling-min-val',
        type=float,
        default=1e-4,
        help='Minimum value to clamp scale to when using bf16 or fp16 quantization.')
    parser.add_argument(
        '--quant-sdpa',
        action='store_true',
        help='Quantize `F.scaled_dot_product_attention` (default: %(default)s)')
    parser.add_argument(
        '--replace-mha',
        action='store_true',
        help='Replace HuggingFace Attention with a quantizable version')
    parser.add_argument(
        '--weight-equalization',
        action='store_true',
        help='Apply weight equalization. Relevant to ReLU based models (e.g. OPT).')
    parser.add_argument(
        '--rotation',
        type=str,
        default=None,
        choices=['fx', 'layerwise', 'fused_no_fx'],
        help='Apply graph rotation equalization')
    parser.add_argument(
        '--rotation-mode',
        default='had',
        choices=['had', 'ort'],
        help=
        'If GraphRotation is enabled, decide how to compute the random rotation matrix that is fully fused. Online or partial rotation will always be Hadamard'
    )
    parser.add_argument(
        '--rotation-orphan-sink',
        action="store_true",
        help=
        'If GraphRotation is enabled, decide wheter to add standalone hadamard matrices for the unfused layers'
    )
    parser.add_argument(
        '--rotation-sdpa-regions',
        action="store_true",
        help='If GraphRotation is enabled, decide wheter to equalize across SDPA')
    parser.add_argument(
        '--act-equalization',
        default=None,
        choices=[None, 'layerwise', 'fx'],
        help='Apply activation equalization (SmoothQuant). Layerwise introduces standalone mul nodes,'
        'while fx merges them whenever possible into previous tensors, which is possible on ReLU based models (e.g. OPT).'
    )
    parser.add_argument(
        '--act-equalization-alpha',
        default=0.5,
        type=float,
        help='If activation equalization is enabled, decide what alpha to use')
    parser.add_argument('--load-awq', type=str, default=None, help="Load the awq search results.")
    parser.add_argument(
        '--export-target',
        default=None,
        choices=[
            None,
            'onnx_qcdq',
            'torch_qcdq',
            'sharded_torchmlir_group_weight',
            'sharded_packed_torchmlir_group_weight'],
        help='Model export.')
    parser.add_argument(
        '--export-prefix',
        type=str,
        default=None,
        help=
        "Path prefix to use for the various export flows. If None, a path will be derived from the model name (default: %(default)s)"
    )
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help="Filename to save checkpoint. If `None`, no checkpoint is saved (default: %(default)s)"
    )
    parser.add_argument(
        "--fuse-sequences",
        action="store_true",
        default=False,
        help=
        "Whether to merge the dataset sequences in case they are shorter than the requested number of samples per sequence. This is useful in case you would like to quantize or evaluate on long sequences (default: %(default)s).",
    )
    parser.add_argument(
        '--learned-round',
        default=None,
        choices=[None, 'linear_round'],
        help='Whether to use learned round. If `None`, RTN is used (default: %(default)s)')
    parser.add_argument(
        '--learned-round-fast-update',
        default=False,
        action="store_true",
        help='Whether to use fast update with learned round. Prototype (default: %(default)s)')
    parser.add_argument(
        '--few-shot-eval',
        action="store_true",
        help='Perform zero_shot evaluation with lm_eval. Default %(default)s)')
    parser.add_argument(
        '--few-shot-compile',
        action="store_true",
        help='Compile during zero_shot evaluation with lm_eval. Default %(default)s)')
    parser.add_argument(
        '--few-shot-zeroshot',
        action="store_true",
        help='Whether to do zero or few shot eval. Default %(default)s)')
    parser.add_argument(
        '--few-shot-limit', type=int, default=None, help='Few shot limit. Default %(default)s)')
    parser.add_argument(
        '--few-shot-tasks',
        default=['arc_challenge', 'arc_easy', 'winogrande', 'piqa'],
        type=str,
        nargs='*',
        help='A list of tasks for zero_shot evaluation. Default: %(default)s')
    parser.set_defaults(**override_defaults)

    return parser.parse_args(args)


def main():
    overrides = override_defaults(sys.argv[1:])
    args = parse_args(sys.argv[1:], override_defaults=overrides)
    quantize_llm(args)


if __name__ == '__main__':
    main()
