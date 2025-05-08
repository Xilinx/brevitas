# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import functools
import pprint
import sys

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
from brevitas.graph import load_quant_model_mode
from brevitas.graph.base import ModuleInstanceTransformTensor
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import LayerwiseActivationRotation
from brevitas.graph.quantize import functional_quantization_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.utils import get_module
from brevitas.nn.quant_sdpa import ScaledDotProductAttention
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.llm.llm_args import create_llm_args_parser
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
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
from brevitas_examples.llm.llm_quant.gpxq import apply_magr
from brevitas_examples.llm.llm_quant.learned_round_utils import apply_learned_round
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.prepare_for_quantize import \
    replace_sdpa_with_quantizable_layers
from brevitas_examples.llm.llm_quant.rotation_optimization import apply_rotation_optimization
from brevitas_examples.llm.llm_quant.rotation_optimization import parse_rotation_optimization_args
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.llm_quant.run_utils import get_fx
from brevitas_examples.llm.llm_quant.svd_quant import apply_svd_quant


def filter_results(results, tasks):
    # filter out what we actually want to track
    eval_results = dict()
    for task_name in tasks:
        # log all result metrics we have for this task
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
    model.config.use_cache = False
    with torch.no_grad():
        new_model, guards = torch._dynamo.export(model)(**calibration_loader[0])
    if hasattr(model, str(torch.nn.functional.scaled_dot_product_attention)):
        m_to_add = getattr(model, str(torch.nn.functional.scaled_dot_product_attention))
        new_model.add_module(str(torch.nn.functional.scaled_dot_product_attention), m_to_add)

    layers_to_expand = []
    if args.rotation is not None:
        for name, _ in new_model.named_modules():
            if any(map(lambda x: x in name, args.rotation_layers_to_expand)):
                layers_to_expand.append(name)

    apply_layernorm_affine_merge(new_model)
    # NOTE: This call breaks ties between the the lm_head and the embedding layer
    new_model, rewriters = apply_layernorm_to_rmsnorm(new_model, return_rewriters=True)
    rewriters = fix_rewriter(rewriters, model, 'weight')

    for r in rewriters:
        r.apply(model)
    new_model = offload_model(new_model)
    eq = GraphRotationEqualization(
        orphan_sink=args.rotation_orphan_sink,
        full_rotation_method=args.rotation_mode,
        return_rewriters=True,
        sdpa_regions=args.rotation_sdpa_regions,
        use_parametrized_rotations=args.optimize_rotations,
        layers_to_expand=layers_to_expand)
    new_model, rewriters = eq.apply(new_model)
    rewriters = fix_rewriter(rewriters, model, 'weight')
    with torch.no_grad():
        for r in rewriters:
            # The weights between model and new_model are tied, so this check prevents
            # rotating the weights twice
            if not isinstance(r, ModuleInstanceTransformTensor):
                model = r.apply(model)
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


def fx_required(args):
    quant_sdpa_fx = args.quant_sdpa and not args.replace_mha
    return True if args.weight_equalization or args.act_equalization == 'fx' or args.rotation == 'fx' or args.ln_affine_merge or args.convert_layernorm_to_rmsnorm or quant_sdpa_fx else False


def quantize_llm(args, extra_args=None):
    validate(args, extra_args)
    set_seed(args.seed)
    if args.export_prefix is None:
        args.export_prefix = f"{args.model.replace('/', '--')}"

    dtype = getattr(torch, args.dtype)

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

    require_fx = fx_required(args)

    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=require_fx and args.export_target is not None,
        device=None)

    validation_loader = get_dataset_for_model(
        args.model,
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="validation",
        seed=args.seed,
        require_fx=require_fx and args.export_target is not None,
        device=None)

    if args.optimize_rotations:
        # Extra arguments should be used as training arguments for rotation optimization
        rot_optimization_args = parse_rotation_optimization_args(extra_args=extra_args)
        # Load the data for rotation optimization
        rot_calibration_loader = get_dataset_for_model(
            args.model,
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            nsamples=args.nsamples_rot_calibration,
            seqlen=args.seqlen,
            split="train",
            seed=args.seed,
            require_fx=require_fx and args.export_target is not None,
            device=None)

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
    elif args.functional_sdpa_quant:
        print("Inserting SDPA quantizable module")
        model = offload_model(model)
        with torch.no_grad(), functional_quantization_mode(model, {torch.nn.functional.scaled_dot_product_attention: ScaledDotProductAttention}):
            model(**calibration_loader[0])
        remove_hooks(model)

    layers_to_expand = []
    if args.rotation is not None:
        for name, _ in model.named_modules():
            if any(map(lambda x: x in name, args.rotation_layers_to_expand)):
                layers_to_expand.append(name)
    if args.rotation == 'fx':
        model = offload_model(model)
        eq = GraphRotationEqualization(
            orphan_sink=args.rotation_orphan_sink,
            full_rotation_method=args.rotation_mode,
            sdpa_regions=args.rotation_sdpa_regions,
            use_parametrized_rotations=args.optimize_rotations,
            layers_to_expand=layers_to_expand)
        model = eq.apply(model)
        remove_hooks(model)
    elif args.rotation == 'layerwise':
        eq = LayerwiseActivationRotation(layers_to_expand=layers_to_expand)
        model = eq.apply(model)
    elif args.rotation == 'fused_no_fx':
        fused_rotation_no_fx(model, calibration_loader, args)

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
        if args.load_checkpoint:
            loader = [calibration_loader[0]]
        else:
            loader = calibration_loader
        apply_act_equalization(
            model, args.act_equalization, loader, alpha=args.act_equalization_alpha)
        print("Act equalization applied.")
        remove_hooks(model)

    if args.magr and not args.load_checkpoint:
        print("Applying MagR...")
        model = offload_model(model)
        apply_magr(
            model,
            calibration_loader,
            create_weight_orig=args.gpxq_create_weight_orig or
            args.gpfq,  # save original weights for GPxQ
            alpha=args.magr_alpha)
        remove_hooks(model)
        print(f"MagR applied.")

    if not args.no_quantize:
        name_blacklist = []
        print("Applying model quantization...")
        # When AWQ is enabled, the scaling_impl_type for the weights needs to be 'stats', as the
        # scaling factor that multiplies the weights is optimized
        weight_scaling_impl_type = 'stats' if (
            args.awq_scale or args.awq_clip) else 'parameter_from_stats'
        linear_input_quant, weight_quant, input_quant, q_scaled_quant, k_transposed_quant, v_quant, attn_output_weights_quant = generate_quantizers(
            dtype=dtype,
            weight_bit_width=args.weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_type=args.weight_quant_type,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            weight_group_dim=args.weight_group_dim,
            weight_scaling_impl_type=weight_scaling_impl_type,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            weight_quant_format=args.weight_quant_format,
            input_bit_width=args.input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_precision=args.input_scale_precision,
            input_scale_type=args.input_scale_type,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            kv_quant_type=args.kv_quant_type,
            kv_quant_granularity=args.kv_quant_granularity,
            input_quant_granularity=args.input_quant_granularity,
            input_group_size=args.input_group_size,
            quantize_input_zero_point=args.quantize_input_zero_point,
            scale_rounding_func_type=args.scale_rounding_func_type,
            quant_attn_mode='sdpa' if (quant_sdpa_fx or args.functional_sdpa_quant) else 'mha',
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
            # Dynamo tracing changes the name of the modules, thus we need this workaround to pick
            # up the last module.
            if require_fx:
                last_node = [node for node in model.graph.nodes if node.op == 'call_module'][-1]
                last_module = get_module(model, last_node.target)
                # In case we have layerwise rotation/equalization, we need to pick the wrapped module
                last_module = last_module.layer if hasattr(last_module, 'layer') else last_module
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
        # Just to be sure
        model.eval()
        # Tie back first/last layer weights in case they got untied
        print("Model quantization applied.")

    if args.awq_scale or args.awq_clip:
        apply_awq(
            model=model,
            tokenizer=tokenizer,
            calibration_loader=calibration_loader,
            args=args,
            auto_scale=args.awq_scale,
            mse_range=args.awq_clip,
        )

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

    # If we are doing functional SDPA quantization, we create the correct context manager,
    # otherwise nullcontext. We would love to avoid the extra indentation level but it doesn't seem easy.
    if args.functional_sdpa_quant:
        quantization_cm = functional_quantization_mode(
            model, {torch.nn.functional.scaled_dot_product_attention: ScaledDotProductAttention})
    else:
        quantization_cm = nullcontext()

    with quantization_cm:
        # We initialize weights scale factor pre-GPTQ
        with torch.no_grad():
            model(**calibration_loader[0])

        if args.compile_ptq:
            for m in model.modules():
                if hasattr(m, 'compile_quant'):
                    m.compile_quant()

        if args.act_calibration and not args.load_checkpoint:
            print("Apply act calibration...")
            apply_calibration(model, calibration_loader)
            print("Act calibration applied.")

        if args.optimize_rotations:
            apply_rotation_optimization(
                model=model,
                tokenizer=tokenizer,
                train_dataset=rot_calibration_loader,
                training_args=rot_optimization_args,
            )
            # Remove hooks from optimization
            remove_hooks(model)
            # Offload model before fusing the rotations
            model = offload_model(model)
            # Fuse rotations with weights
            model = fuse_parametrizations(model)

        if args.svd_quant:
            print("Apply SVDQuant...")
            remove_hooks(model)
            model = apply_svd_quant(
                model,
                blacklist=None,
                rank=args.svd_quant_rank,
                iters=args.svd_quant_iters,
                dtype=torch.float32)
            model = offload_model(model)
            with torch.no_grad():
                model(**calibration_loader[0])
            print("SVDQuant applied.")

        if args.learned_round:
            print("Applying learned round...")
            if args.load_checkpoint:
                iters = 1
                loader = [calibration_loader[0]]
            else:
                iters = args.learned_round_iters
                loader = calibration_loader
            remove_hooks(model)
            apply_learned_round(
                model,
                loader,
                iters=iters,
                block_name_attribute=args.gpxq_block_name,
                learn_scale=args.learned_round_scale,
                scale_optimizer_class='sgd',
                optimizer_kwargs={'lr': args.learned_round_lr},
                scale_optimizer_kwargs={
                    'lr': args.learned_round_scale_lr,
                    'momentum': args.learned_round_scale_momentum},
                fast_update=args.learned_round_fast_update)
            print("Learned round applied.")
            model = offload_model(model)

        if args.load_checkpoint:
            remove_hooks(model)
            with load_quant_model_mode(model):
                model.load_state_dict(torch.load(args.checkpoint_name, map_location='cpu'))
            model = offload_model(model)

        if args.gptq and not args.load_checkpoint:
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

        if args.gpfq and not args.load_checkpoint:
            print("Applying GPFQ...")
            apply_gpfq(
                model,
                calibration_loader,
                act_order=args.gpxq_act_order,
                block_name=args.gpxq_block_name,
                max_accumulator_bit_width=args.gpxq_max_accumulator_bit_width,
                max_accumulator_tile_size=args.gpxq_max_accumulator_tile_size)
            print("GPFQ applied.")

        if args.bias_corr and not args.load_checkpoint:
            print("Applying bias correction...")
            apply_bias_correction(model, calibration_loader)
            print("Bias correction applied.")

        # We restore the original behaviour of the post-forward.
        for k, v in dict_hooks.items():
            k._hf_hook.post_forward = v

        if args.eval and not args.no_quantize:

            print("Model eval...")
            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**calibration_loader[0])
                quant_ppl = compute_perplexity(
                    model, validation_loader, context_length=args.seqlen // 2, tokenizer=tokenizer)
            print(f"Quantized perplexity ({args.dataset}): {quant_ppl:.3f}")

        few_shot_eval_results = dict()
        if args.few_shot_eval == 'lm_eval':
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**calibration_loader[0])

                wrapped_model = HFLM(
                    pretrained=model, add_bos_token=True)  # need to wrap for LLM eval
                few_shot_eval_results = evaluator.simple_evaluate(
                    model=wrapped_model,
                    model_args=None,
                    tasks=list(args.few_shot_tasks),
                    device='cuda:0',
                    limit=args.few_shot_limit,
                    num_fewshot=0 if args.few_shot_zeroshot else None,
                    log_samples=False,
                    batch_size=None,
                    verbosity="ERROR")
            few_shot_eval_results = filter_results(few_shot_eval_results, args.few_shot_tasks)
            print("Few shot eval results")
            pprint.pprint(few_shot_eval_results)
        elif args.few_shot_eval == 'lighteval':
            from accelerate import Accelerator
            from accelerate import InitProcessGroupKwargs
            from huggingface_hub import constants
            from lighteval.logging.evaluation_tracker import EvaluationTracker
            from lighteval.models.transformers.transformers_model import TransformersModelConfig
            from lighteval.pipeline import ParallelismManager
            from lighteval.pipeline import Pipeline
            from lighteval.pipeline import PipelineParameters
            from lighteval.utils.utils import EnvConfig

            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**calibration_loader[0])
                remove_hooks(model)
                # expects a list
                few_shot_tasks = ",".join(args.few_shot_tasks)
                accelerator = Accelerator(
                    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
                evaluation_tracker = EvaluationTracker(output_dir="./results", save_details=True)
                pipeline_params = PipelineParameters(
                    launcher_type=ParallelismManager.ACCELERATE,
                    env_config=EnvConfig(cache_dir=constants.HF_HUB_CACHE),
                    override_batch_size=args.few_shot_override_batch_size)
                model_config = TransformersModelConfig(
                    pretrained=args.model,
                    dtype=dtype,
                    model_parallel=True,
                    accelerator=accelerator)
                pipeline = Pipeline(
                    tasks=few_shot_tasks,
                    pipeline_parameters=pipeline_params,
                    evaluation_tracker=evaluation_tracker,
                    model=model,
                    config=model_config)
                pipeline.evaluate()
            few_shot_eval_results = pipeline.get_results()
            few_shot_eval_results = filter_results(
                few_shot_eval_results, list(few_shot_eval_results["results"].keys()))
            pprint.pprint(few_shot_eval_results)
        remove_hooks(model)
        if args.checkpoint_name is not None and not args.load_checkpoint:
            print(f"Saving checkpoint to {args.checkpoint_name}")
            torch.save(model.state_dict(), args.checkpoint_name)

        if args.export_target:
            print(f"Export to {args.export_target}")
            # Currently we always export on CPU with a float32 container to avoid float16 CPU errors
            model = model.to(dtype=torch.float32)
            model_export(model, calibration_loader[0], args)

    return {"float_ppl": float_ppl, "quant_ppl": quant_ppl, **few_shot_eval_results}, model


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
    parser = create_llm_args_parser()
    if len(override_defaults) > 0:
        # Retrieve keys that are known to the parser
        parser_keys = set(map(lambda action: action.dest, parser._actions))
        # Extract the entries in override_defaults that correspond to keys not known to the parser
        extra_args_keys = [key for key in override_defaults.keys() if key not in parser_keys]
        # Remove all the keys in override_defaults that are unknown to the parser and, instead,
        # include them in args, as if they were passed as arguments to the command line.
        # This prevents the keys of HF TrainingArguments from being added as arguments to the parser.
        # Consequently, they will be part of the second value returned by parse_known_args (thus being
        # used as extra_args in quantize_llm)
        for key in extra_args_keys:
            args += [f"--{key}", str(override_defaults[key])]
            del override_defaults[key]
    parser.set_defaults(**override_defaults)
    return parser.parse_known_args(args)


def main():
    overrides = override_defaults(sys.argv[1:])
    args, extra_args = parse_args(sys.argv[1:], override_defaults=overrides)
    quantize_llm(args, extra_args)


if __name__ == '__main__':
    main()
