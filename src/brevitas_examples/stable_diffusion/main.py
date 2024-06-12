"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from datetime import datetime
from functools import partial
import json
import os
import time
import warnings

from dependencies import value
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttnProcessor
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from brevitas.core.stats.stats_op import NegativeMinOrZero
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.export.torch.qcdq.manager import TorchQCDQManager
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import load_quant_model_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import StatsOp
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.utils.torch_utils import KwargsForwardHook
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.stable_diffusion.mlperf_evaluation.accuracy import compute_mlperf_fid
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_2_1_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_XL_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.export import export_onnx
from brevitas_examples.stable_diffusion.sd_quant.export import export_quant_params
from brevitas_examples.stable_diffusion.sd_quant.export import export_torch_export
from brevitas_examples.stable_diffusion.sd_quant.modules import QuantAttention
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_latents
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_21_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_xl_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import unet_input_shape

TEST_SEED = 123456
torch.manual_seed(TEST_SEED)

NEGATIVE_PROMPTS = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"]

CALIBRATION_PROMPTS = [
    'A man in a space suit playing a guitar, inspired by Cyril Rolando, highly detailed illustration, full color illustration, very detailed illustration, dan mumford and alex grey style',
    'a living room, bright modern Scandinavian style house, large windows, magazine photoshoot, 8k, studio lighting',
    'cute rabbit in a spacesuit',
    'minimalistic plolygon geometric car in brutalism warehouse, Rick Owens']

TESTING_PROMPTS = [
    'batman, cute modern disney style, Pixar 3d portrait, ultra detailed, gorgeous, 3d zbrush, trending on dribbble, 8k render',
    'A beautiful stack of rocks sitting on top of a beach, a picture, red black white golden colors, chakras, packshot, stock photo',
    'A painting of a fish on a black background, a digital painting, by Jason Benjamin, colorful vector illustration, mixed media style illustration, epic full color illustration, mascot illustration',
    'close up photo of a rabbit, forest in spring, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot'
]


def load_calib_prompts(calib_data_path, sep="\t"):
    df = pd.read_csv(calib_data_path, sep=sep)
    lst = df["caption"].tolist()
    return lst


def run_test_inference(
        pipe,
        resolution,
        prompts,
        seeds,
        output_path,
        device,
        dtype,
        use_negative_prompts,
        guidance_scale,
        name_prefix=''):
    images = dict()
    with torch.no_grad():
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        test_latents = generate_latents(seeds, device, dtype, unet_input_shape(resolution))
        neg_prompts = NEGATIVE_PROMPTS * len(seeds) if use_negative_prompts else []
        for prompt in prompts:
            prompt_images = pipe([prompt] * len(seeds),
                                 latents=test_latents,
                                 negative_prompt=neg_prompts,
                                 guidance_scale=guidance_scale).images
            images[prompt] = prompt_images

        i = 0
        for prompt, prompt_images in images.items():
            for image in prompt_images:
                file_path = os.path.join(output_path, f"{name_prefix}{i}.png")
                print(f"Saving to {file_path}")
                image.save(file_path)
                i += 1
    return images


def run_val_inference(
        pipe,
        resolution,
        prompts,
        seeds,
        device,
        dtype,
        use_negative_prompts,
        guidance_scale,
        total_steps,
        test_latents=None):
    with torch.no_grad():

        if test_latents is None:
            test_latents = generate_latents(seeds[0], device, dtype, unet_input_shape(resolution))

        neg_prompts = NEGATIVE_PROMPTS if use_negative_prompts else []
        for prompt in tqdm(prompts):
            # We don't want to generate any image, so we return only the latent encoding pre VAE
            pipe(
                prompt,
                negative_prompt=neg_prompts[0],
                latents=test_latents,
                output_type='latent',
                guidance_scale=guidance_scale,
                num_inference_steps=total_steps)


def main(args):

    dtype = getattr(torch, args.dtype)

    calibration_prompts = CALIBRATION_PROMPTS
    if args.calibration_prompt_path is not None:
        calibration_prompts = load_calib_prompts(args.calibration_prompt_path)
    prompts = list()
    for i, v in enumerate(calibration_prompts):
        if i == args.calibration_prompt:
            break
        prompts.append(v)
    calibration_prompts = prompts

    latents = None
    if args.path_to_latents is not None:
        latents = torch.load(args.path_to_latents).to(torch.float16)
    else:
        warnings.warn("No latent provided. Will generate some using pre-set seeds")

    # Create output dir. Move to tmp if None
    ts = datetime.fromtimestamp(time.time())
    str_ts = ts.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f'{str_ts}')
    os.mkdir(output_dir)
    print(f"Saving results in {output_dir}")

    # Dump args to json
    with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # Extend seeds based on batch_size
    test_seeds = [TEST_SEED] + [TEST_SEED + i for i in range(1, args.batch_size)]

    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    print(f"Model loaded from {args.model}.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    if args.prompt > 0 and not args.use_mlperf_inference:
        print(f"Running inference with prompt ...")
        prompts = []
        for i, v in enumerate(TESTING_PROMPTS):
            if i == args.prompt:
                break
            prompts.append(v)
        float_images = run_test_inference(
            pipe,
            args.resolution,
            prompts,
            test_seeds,
            output_dir,
            args.device,
            dtype,
            guidance_scale=args.guidance_scale,
            use_negative_prompts=args.use_negative_prompts,
            name_prefix='float_')

    # Detect Stable Diffusion XL pipeline
    is_sd_xl = isinstance(pipe, StableDiffusionXLPipeline)

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    for name, _ in pipe.unet.named_modules():
        if 'time_emb' in name and not args.quantize_time_emb:
            blacklist.append(name.split('.')[-1])
        if 'conv_in' in name and not args.quantize_conv_in:
            blacklist.append(name.split('.')[-1])
    print(f"Blacklisted layers: {blacklist}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in pipe.unet.modules():
        if hasattr(m, 'lora_layer') and m.lora_layer is not None:
            raise RuntimeError("LoRA layers should be fused in before calling into quantization.")

    if args.activation_equalization:
        pipe.set_progress_bar_config(disable=True)
        with activation_equalization_mode(pipe.unet,
                                          alpha=args.act_eq_alpha,
                                          layerwise=True,
                                          blacklist_layers=blacklist,
                                          add_mul_node=True):
            # Workaround to expose `in_features` attribute from the Hook Wrapper
            for m in pipe.unet.modules():
                if isinstance(m, KwargsForwardHook) and hasattr(m.module, 'in_features'):
                    m.in_features = m.module.in_features
            total_steps = args.calibration_steps
            if args.dry_run or args.load_checkpoint is not None:
                calibration_prompts = [calibration_prompts[0]]
                total_steps = 1
            run_val_inference(
                pipe,
                args.resolution,
                calibration_prompts,
                test_seeds,
                args.device,
                dtype,
                total_steps=total_steps,
                use_negative_prompts=args.use_negative_prompts,
                test_latents=latents,
                guidance_scale=args.guidance_scale)

        # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
        for m in pipe.unet.modules():
            if isinstance(m, EqualizedModule) and hasattr(m.layer, 'in_features'):
                m.in_features = m.layer.in_features

    # Quantize model
    if args.quantize:

        @value
        def weight_bit_width(module):
            if isinstance(module, nn.Linear):
                return args.linear_weight_bit_width
            elif isinstance(module, nn.Conv2d):
                return args.conv_weight_bit_width
            else:
                raise RuntimeError(f"Module {module} not supported.")

        @value
        def input_bit_width(module):
            if isinstance(module, nn.Linear):
                return args.linear_input_bit_width
            elif isinstance(module, nn.Conv2d):
                return args.conv_input_bit_width
            elif isinstance(module, QuantIdentity):
                return args.quant_identity_bit_width
            else:
                raise RuntimeError(f"Module {module} not supported.")

        input_kwargs = dict()
        if args.input_scale_stats_op == 'minmax':

            @value
            def input_scale_stats_type():
                if args.input_quant_type == 'asym':
                    input_scaling_stats_op = StatsOp.MIN_MAX
                else:
                    input_scaling_stats_op = StatsOp.MAX
                return input_scaling_stats_op

            input_kwargs['scaling_stats_op'] = input_scale_stats_type

        if args.input_zp_stats_op == 'minmax':

            @value
            def input_zp_stats_type():
                if args.input_quant_type == 'asym':
                    zero_point_stats_impl = NegativeMinOrZero
                    return zero_point_stats_impl

            input_kwargs['zero_point_stats_impl'] = input_zp_stats_type

        print("Applying model quantization...")
        quantizers = generate_quantizers(
            dtype=dtype,
            device=args.device,
            weight_bit_width=weight_bit_width,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            quantize_input_zero_point=args.quantize_input_zero_point,
            input_bit_width=input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_type=args.input_scale_type,
            input_scale_precision=args.input_scale_precision,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            use_ocp=args.use_ocp,
            input_kwargs=input_kwargs)

        layer_map = generate_quant_maps(
            *quantizers, dtype, args.device, args.input_quant_format, False)
        if not args.quantize_linear_conv:
            for _, (quant_class, quant_args) in layer_map.items():
                for k, v in quant_args.items():
                    if 'quant' in k:
                        quant_args[k] = None

        linear_qkwargs = layer_map[torch.nn.Linear][1]
        linear_qkwargs[
            'input_quant'] = None if args.linear_input_bit_width is None else linear_qkwargs[
                'input_quant']
        linear_qkwargs[
            'weight_quant'] = None if args.linear_weight_bit_width == 0 else linear_qkwargs[
                'weight_quant']
        layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], linear_qkwargs)

        conv_qkwargs = layer_map[torch.nn.Conv2d][1]
        conv_qkwargs['input_quant'] = None if args.conv_input_bit_width is None else conv_qkwargs[
            'input_quant']
        conv_qkwargs['weight_quant'] = None if args.conv_weight_bit_width == 0 else conv_qkwargs[
            'weight_quant']
        layer_map[torch.nn.Conv2d] = (layer_map[torch.nn.Conv2d][0], conv_qkwargs)

        if args.quantize_sdp_1 or args.quantize_sdp_2:
            float_sdpa_quantizers = generate_quantizers(
                dtype=dtype,
                device=args.device,
                weight_bit_width=weight_bit_width,
                weight_quant_format='e4m3',
                weight_quant_type='sym',
                weight_param_method=args.weight_param_method,
                weight_scale_precision=args.weight_scale_precision,
                weight_quant_granularity=args.weight_quant_granularity,
                weight_group_size=args.weight_group_size,
                quantize_weight_zero_point=args.quantize_weight_zero_point,
                quantize_input_zero_point=args.quantize_input_zero_point,
                input_bit_width=input_bit_width,
                input_quant_format='e4m3',
                input_scale_type=args.input_scale_type,
                input_scale_precision=args.input_scale_precision,
                input_param_method=args.input_param_method,
                input_quant_type='sym',
                input_quant_granularity=args.input_quant_granularity,
                use_ocp=args.use_ocp,
                input_kwargs=input_kwargs)
            input_quant = float_sdpa_quantizers[0]
            input_quant = input_quant.let(**{'bit_width': args.linear_output_bit_width})
            if args.quantize_sdp_2:
                rewriter = ModuleToModuleByClass(
                    Attention,
                    QuantAttention,
                    softmax_output_quant=input_quant,
                    query_dim=lambda module: module.to_q.in_features,
                    dim_head=lambda module: int(1 / (module.scale ** 2)),
                    processor=AttnProcessor(),
                    is_equalized=args.activation_equalization)
                import brevitas.config as config
                config.IGNORE_MISSING_KEYS = True
                pipe.unet = rewriter.apply(pipe.unet)
                config.IGNORE_MISSING_KEYS = False
                pipe.unet = pipe.unet.to(args.device)
                pipe.unet = pipe.unet.to(dtype)
            quant_kwargs = layer_map[torch.nn.Linear][1]
            what_to_quantize = []
            if args.quantize_sdp_1:
                what_to_quantize.extend(['to_q', 'to_k'])
            if args.quantize_sdp_2:
                what_to_quantize.extend(['to_v'])
            quant_kwargs['output_quant'] = lambda module, name: input_quant if any(ending in name for ending in what_to_quantize) else None
            layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], quant_kwargs)

        pipe.unet = layerwise_quantize(
            model=pipe.unet, compute_layer_map=layer_map, name_blacklist=blacklist)

        print("Model quantization applied.")

        skipped_layers = []
        for name, module in pipe.unet.named_modules():
            if 'time_emb' in name and not args.quantize_input_time_emb:
                if hasattr(module, 'input_quant'):
                    module.input_quant.quant_injector = module.input_quant.quant_injector.let(
                        **{'quant_type': QuantType.FP})
                    module.input_quant.init_tensor_quant()
                    skipped_layers.append(name)
            if 'conv_in' in name and not args.quantize_input_conv_in:
                if hasattr(module, 'input_quant'):
                    module.input_quant.quant_injector = module.input_quant.quant_injector.let(
                        **{'quant_type': QuantType.FP})
                    module.input_quant.init_tensor_quant()
                    skipped_layers.append(name)
        print(f"Skipped input quantization for layers: {skipped_layers}")

        pipe.set_progress_bar_config(disable=True)

        if args.dry_run or args.load_checkpoint:
            with torch.no_grad():
                run_val_inference(
                    pipe,
                    args.resolution, [calibration_prompts[0]],
                    test_seeds,
                    args.device,
                    dtype,
                    total_steps=1,
                    use_negative_prompts=args.use_negative_prompts,
                    test_latents=latents,
                    guidance_scale=args.guidance_scale)

        if args.load_checkpoint is not None:
            with load_quant_model_mode(pipe.unet):
                pipe = pipe.to('cpu')
                pipe.unet.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
                pipe = pipe.to(args.device)
        elif not args.dry_run:
            if (args.linear_input_bit_width is not None or
                    args.conv_input_bit_width is not None) and args.input_scale_type == 'static':
                print("Applying activation calibration")
                with torch.no_grad(), calibration_mode(pipe.unet):
                    run_val_inference(
                        pipe,
                        args.resolution,
                        calibration_prompts,
                        test_seeds,
                        args.device,
                        dtype,
                        total_steps=args.calibration_steps,
                        use_negative_prompts=args.use_negative_prompts,
                        test_latents=latents,
                        guidance_scale=args.guidance_scale)

            if args.gptq:
                print("Applying GPTQ. It can take several hours")
                with torch.no_grad(), gptq_mode(pipe.unet,
                            create_weight_orig=False,
                            use_quant_activations=False,
                            return_forward_output=True,
                            act_order=True) as gptq:
                    for _ in tqdm(range(gptq.num_layers)):
                        run_val_inference(
                            pipe,
                            args.resolution,
                            calibration_prompts,
                            test_seeds,
                            args.device,
                            dtype,
                            total_steps=args.calibration_steps,
                            use_negative_prompts=args.use_negative_prompts,
                            test_latents=latents,
                            guidance_scale=args.guidance_scale)
                        gptq.update()
                        torch.cuda.empty_cache()
            if args.bias_correction:
                print("Applying bias correction")
                with bias_correction_mode(pipe.unet):
                    run_val_inference(
                        pipe,
                        args.resolution,
                        calibration_prompts,
                        test_seeds,
                        args.device,
                        dtype,
                        total_steps=args.calibration_steps,
                        use_negative_prompts=args.use_negative_prompts,
                        test_latents=latents,
                        guidance_scale=args.guidance_scale)

    if args.checkpoint_name is not None and args.load_checkpoint is None:
        torch.save(pipe.unet.state_dict(), os.path.join(output_dir, args.checkpoint_name))

    # Perform inference
    if args.prompt > 0 and not args.dry_run:
        # with brevitas_proxy_inference_mode(pipe.unet):
        if args.use_mlperf_inference:
            print(f"Computing accuracy with MLPerf pipeline")
            compute_mlperf_fid(args.model, args.path_to_coco, pipe, args.prompt, output_dir)
        else:
            print(f"Computing accuracy on default prompt")
            prompts = list()
            for i, v in enumerate(TESTING_PROMPTS):
                if i == args.prompt:
                    break
                prompts.append(v)
            quant_images = run_test_inference(
                pipe,
                args.resolution,
                prompts,
                test_seeds,
                output_dir,
                args.device,
                dtype,
                use_negative_prompts=args.use_negative_prompts,
                guidance_scale=args.guidance_scale,
                name_prefix='quant_')

            float_images_values = float_images.values()
            float_images_values = [x for x_nested in float_images_values for x in x_nested]
            float_images_values = torch.tensor([np.array(image) for image in float_images_values])
            float_images_values = float_images_values.permute(0, 3, 1, 2)

            quant_images_values = quant_images.values()
            quant_images_values = [x for x_nested in quant_images_values for x in x_nested]
            quant_images_values = torch.tensor([np.array(image) for image in quant_images_values])
            quant_images_values = quant_images_values.permute(0, 3, 1, 2)

            fid = FrechetInceptionDistance(normalize=False)
            fid.update(float_images_values, real=True)
            fid.update(quant_images_values, real=False)
            print(f"FID: {float(fid.compute())}")

    if args.export_target:
        # Move to cpu and to float32 to enable CPU export
        if not (dtype == torch.float16 and args.export_cuda_float16):
            pipe.unet.to('cpu').to(dtype)
        pipe.unet.eval()
        device = next(iter(pipe.unet.parameters())).device
        dtype = next(iter(pipe.unet.parameters())).dtype

        # Define tracing input
        if is_sd_xl:
            generate_fn = generate_unet_xl_rand_inputs
            shape = SD_XL_EMBEDDINGS_SHAPE
        else:
            generate_fn = generate_unet_21_rand_inputs
            shape = SD_2_1_EMBEDDINGS_SHAPE
        trace_inputs = generate_fn(
            embedding_shape=shape,
            unet_input_shape=unet_input_shape(args.resolution),
            device=device,
            dtype=dtype)

        if args.export_target == 'onnx':
            if args.weight_quant_granularity == 'per_group':
                export_manager = BlockQuantProxyLevelManager
            else:
                export_manager = StdQCDQONNXManager
                export_manager.change_weight_export(export_weight_q_node=args.export_weight_q_node)
            export_onnx(pipe, trace_inputs, output_dir, export_manager)
        if args.export_target == 'torch':
            if args.weight_quant_granularity == 'per_group':
                export_manager = BlockQuantProxyLevelManager
            else:
                export_manager = TorchQCDQManager
                export_manager.change_weight_export(export_weight_q_node=args.export_weight_q_node)
            export_torch_export(pipe, trace_inputs, output_dir, export_manager)
        if args.export_target == 'param_dump':
            export_quant_params(pipe, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Diffusion quantization')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='/scratch/hf_models/stable-diffusion-2-1-base',
        help='Path or name of the model.')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0', help='Target device for quantized model.')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=2,
        help='How many seeds to use for each image during validation. Default: 2')
    parser.add_argument(
        '--prompt',
        type=int,
        default=4,
        help='Number of prompt to use for testing. Default: 4. Max: 4')
    parser.add_argument(
        '--calibration-prompt',
        type=int,
        default=2,
        help='Number of prompt to use for calibration. Default: 2')
    parser.add_argument(
        '--calibration-prompt-path', type=str, default=None, help='Path to calibration prompt')
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help=
        'Name to use to store the checkpoint in the output dir. If not provided, no checkpoint is saved.'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load. If provided, PTQ techniques are skipped.')
    parser.add_argument(
        '--path-to-latents',
        type=str,
        default=None,
        help=
        'Load pre-defined latents. If not provided, they are generated based on an internal seed.')
    parser.add_argument(
        '--path-to-coco',
        type=str,
        default=None,
        help=
        'Path to MLPerf compliant Coco dataset. Used when the --use-mlperf flag is set. Default: None'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Resolution along height and width dimension. Default: 512.')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale.')
    parser.add_argument(
        '--calibration-steps', type=float, default=8, help='Steps used during calibration')
    add_bool_arg(
        parser,
        'output-path',
        str_true=True,
        default='.',
        help='Path where to generate output folder.')
    add_bool_arg(parser, 'quantize', default=True, help='Toggle quantization. Default: Enabled')
    add_bool_arg(
        parser,
        'activation-equalization',
        default=False,
        help='Toggle Activation Equalization. Default: Disabled')
    add_bool_arg(parser, 'gptq', default=False, help='Toggle gptq. Default: Disabled')
    add_bool_arg(
        parser, 'bias-correction', default=True, help='Toggle bias-correction. Default: Enabled')
    parser.add_argument(
        '--dtype',
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Model Dtype, choices are float32, float16, bfloat16. Default: float16')
    add_bool_arg(
        parser,
        'attention-slicing',
        default=False,
        help='Enable attention slicing. Default: Disabled')
    parser.add_argument(
        '--export-target',
        type=str,
        default='',
        choices=['', 'torch', 'onnx', 'param_dump'],
        help='Target export flow.')
    add_bool_arg(
        parser,
        'export-weight-q-node',
        default=False,
        help=
        'Enable export of floating point weights + QDQ rather than integer weights + DQ. Default: Disabled'
    )
    parser.add_argument(
        '--conv-weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
    parser.add_argument(
        '--linear-weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
    parser.add_argument(
        '--quant-identity-bit-width', type=int, default=8, help='Weight bit width. Default: None')
    parser.add_argument(
        '--conv-input-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (not quantized)')
    parser.add_argument(
        '--act-eq-alpha',
        type=float,
        default=0.9,
        help='Alpha for activation equalization. Default: 0.9')
    parser.add_argument(
        '--linear-input-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (not quantized).')
    parser.add_argument(
        '--linear-output-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (not quantized).')
    parser.add_argument(
        '--weight-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help='How scales/zero-point are determined. Default: stats.')
    parser.add_argument(
        '--input-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help='How scales/zero-point are determined. Default: stats.')
    parser.add_argument(
        '--input-scale-stats-op',
        type=str,
        default='minmax',
        choices=['minmax', 'percentile'],
        help='Define what statics op to use for input scale. Default: minmax.')
    parser.add_argument(
        '--input-zp-stats-op',
        type=str,
        default='minmax',
        choices=['minmax', 'percentile'],
        help='Define what statics op to use for input zero point. Default: minmax.')
    parser.add_argument(
        '--weight-scale-precision',
        type=str,
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help='Whether scale is a float value or a po2. Default: float_scale.')
    parser.add_argument(
        '--input-scale-precision',
        type=str,
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help='Whether scale is a float value or a po2. Default: float_scale.')
    parser.add_argument(
        '--weight-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Weight quantization type. Default: asym.')
    parser.add_argument(
        '--input-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Input quantization type. Default: asym.')
    parser.add_argument(
        '--weight-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. Default: int.')
    parser.add_argument(
        '--input-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Input quantization type. Either int or eXmY, with X+Y==input_bit_width-1. Default: int.')
    parser.add_argument(
        '--weight-quant-granularity',
        type=str,
        default='per_channel',
        choices=['per_channel', 'per_tensor', 'per_group'],
        help='Granularity for scales/zero-point of weights. Default: per_channel.')
    parser.add_argument(
        '--input-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor'],
        help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
    parser.add_argument(
        '--input-scale-type',
        type=str,
        default='static',
        choices=['static', 'dynamic'],
        help='Whether to do static or dynamic input quantization. Default: static.')
    parser.add_argument(
        '--weight-group-size',
        type=int,
        default=16,
        help='Group size for per_group weight quantization. Default: 16.')
    add_bool_arg(
        parser,
        'quantize-weight-zero-point',
        default=True,
        help='Quantize weight zero-point. Default: Enabled')
    add_bool_arg(
        parser,
        'quantize-input-zero-point',
        default=False,
        help='Quantize input zero-point. Default: Enabled')
    add_bool_arg(
        parser, 'export-cuda-float16', default=False, help='Export FP16 on CUDA. Default: Disabled')
    add_bool_arg(
        parser,
        'use-mlperf-inference',
        default=False,
        help='Evaluate FID score with MLPerf pipeline. Default: False')
    add_bool_arg(
        parser,
        'use-ocp',
        default=True,
        help='Use OCP format for float quantization. Default: True')
    add_bool_arg(
        parser,
        'use-negative-prompts',
        default=True,
        help='Use negative prompts during generation/calibration. Default: Enabled')
    add_bool_arg(
        parser,
        'dry-run',
        default=False,
        help='Generate a quantized model without any calibration. Default: Disabled')
    add_bool_arg(
        parser,
        'quantize-time-emb',
        default=False,
        help='Quantize time embedding layers. Default: False')
    add_bool_arg(
        parser, 'quantize-conv-in', default=True, help='Quantize first conv layer. Default: True')
    add_bool_arg(
        parser,
        'quantize-input-time-emb',
        default=False,
        help='Quantize input to time embedding layers. Default: Disabled')
    add_bool_arg(
        parser,
        'quantize-input-conv-in',
        default=True,
        help='Quantize input to first conv layer. Default: Enabled')
    add_bool_arg(parser, 'quantize-sdp-1', default=False, help='Quantize SDP. Default: Disabled')
    add_bool_arg(parser, 'quantize-sdp-2', default=False, help='Quantize SDP. Default: Disabled')
    add_bool_arg(
        parser,
        'quantize-linear-conv',
        default=True,
        help='Perform quantization through layer replacement of linear/conv. Default: Enabled')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
