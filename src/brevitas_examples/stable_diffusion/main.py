"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from datetime import datetime
import json
import math
import os
import time
import warnings

from dependencies import value
import diffusers
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import packaging
import packaging.version
import pandas as pd
from safetensors.torch import save_file
import torch
from torch import nn
# FID Computation can be performed with several different libraries.
# Each will produce slightly different but valid results
from torchmetrics.image.fid import FrechetInceptionDistance

torch._dynamo.config.force_parameter_static_shapes = False
try:
    from cleanfid import fid as cleanfid
except:
    cleanfid = None
import torchvision.io as image_io
from tqdm import tqdm

from brevitas.core.stats.stats_op import NegativeMinOrZero
from brevitas.export.inference import quant_inference_mode
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import load_quant_model_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.inject.enum import StatsOp
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas.utils.torch_utils import KwargsForwardHook
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.common.svd_quant import ErrorCorrectedModule
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.svd_quant import apply_svd_quant
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_XL_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.export import export_onnx
from brevitas_examples.stable_diffusion.sd_quant.export import export_quant_params
from brevitas_examples.stable_diffusion.sd_quant.nn import AttnProcessor
from brevitas_examples.stable_diffusion.sd_quant.nn import FusedFluxAttnProcessor2_0
from brevitas_examples.stable_diffusion.sd_quant.nn import QuantAttention
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_latents
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_21_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_xl_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import unet_input_shape

diffusers_version = packaging.version.parse(diffusers.__version__)
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


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            image = image_io.read_image(img_path)
            images.append(image)
    return images


def load_calib_prompts(calib_data_path, sep="\t"):
    df = pd.read_csv(calib_data_path, sep=sep)
    lst = df["caption"].tolist()
    return lst


def get_denoising_network(pipe):
    if hasattr(pipe, 'unet'):
        return pipe.unet
    elif hasattr(pipe, 'transformer'):
        return pipe.transformer
    else:
        raise ValueError


def run_test_inference(
        pipe,
        prompts,
        seeds,
        device,
        use_negative_prompts,
        guidance_scale,
        resolution,
        output_path=None,
        subfolder='',
        inference_steps=50,
        deterministic=True):

    with torch.no_grad():
        full_dir = os.path.join(output_path, subfolder)
        generator = torch.Generator(device).manual_seed(0) if deterministic else None
        if output_path is not None:
            os.makedirs(full_dir, exist_ok=True)

        neg_prompts = NEGATIVE_PROMPTS * len(seeds) if use_negative_prompts else []
        i = 0
        for prompt in tqdm(prompts):
            prompt_images = pipe([prompt] * len(seeds),
                                 negative_prompt=neg_prompts,
                                 height=resolution,
                                 width=resolution,
                                 guidance_scale=guidance_scale,
                                 num_inference_steps=inference_steps,
                                 generator=generator).images

            if output_path is not None:
                for image in prompt_images:
                    file_path = os.path.join(full_dir, f"image_{i}.png")
                    # print(f"Saving to {file_path}")
                    image.save(file_path)
                    i += 1


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
        test_latents=None,
        output_type='latent',
        deterministic=True,
        is_unet=True,
        batch=1):
    with torch.no_grad():

        extra_kwargs = {}

        if is_unet and test_latents is not None:
            extra_kwargs['latents'] = test_latents

        generator = torch.Generator(device).manual_seed(0) if deterministic else None
        neg_prompts = NEGATIVE_PROMPTS if use_negative_prompts else []
        for index in tqdm(range(0, len(prompts), batch)):
            curr_prompts = prompts[index:index + batch]

            pipe(
                curr_prompts,
                negative_prompt=neg_prompts * len(curr_prompts),
                output_type=output_type,
                guidance_scale=guidance_scale,
                height=resolution,
                width=resolution,
                num_inference_steps=total_steps,
                generator=generator,
                **extra_kwargs)


def collect_vae_calibration(pipe, calibration, test_seeds, dtype, latents, args):
    new_calibration = []

    def collect_inputs(*input_args, **input_kwargs):
        input_args = tuple([
            input_arg.cpu() if isinstance(input_arg, torch.Tensor) else input_arg
            for input_arg in input_args])
        input_kwargs = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v) for (k, v) in input_kwargs.items()}
        new_calibration.append((input_args, input_kwargs))

    original_vae_decode = pipe.vae.decode
    pipe.vae.decode = hooked_on_a_function(pipe.vae.decode, collect_inputs)
    run_val_inference(
        pipe,
        args.resolution,
        calibration,
        test_seeds,
        args.device,
        dtype,
        deterministic=args.deterministic,
        total_steps=args.calibration_steps,
        use_negative_prompts=args.use_negative_prompts,
        test_latents=latents,
        guidance_scale=args.guidance_scale,
        output_type='pil')

    pipe.vae.decode = original_vae_decode
    return new_calibration


def main(args):

    dtype = getattr(torch, args.dtype)

    calibration_prompts = CALIBRATION_PROMPTS
    if args.calibration_prompt_path is not None:
        calibration_prompts = load_calib_prompts(args.calibration_prompt_path)

    assert args.calibration_prompt <= len(calibration_prompts) , f"Only {len(calibration_prompts)} prompts are available"
    calibration_prompts = calibration_prompts[:args.calibration_prompt]

    latents = None
    if args.path_to_latents is not None:
        latents = torch.load(args.path_to_latents).to(dtype)

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

    is_flux = 'flux' in args.model.lower()
    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")

    extra_kwargs = {}
    if not is_flux:
        variant_dict = {torch.float16: 'fp16', torch.bfloat16: 'bf16'}
        extra_kwargs = {'variant': variant_dict.get(dtype, None)}

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, **extra_kwargs)

    # Detect if is unet-based pipeline
    is_unet = hasattr(pipe, 'unet')
    # Detect Stable Diffusion XL pipeline
    is_sd_xl = isinstance(pipe, StableDiffusionXLPipeline)

    if is_unet:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.vae.config.force_upcast = True

    denoising_network = get_denoising_network(pipe)
    if args.share_qkv_quant:
        if hasattr(pipe, 'fuse_qkv_projections'):
            pipe.fuse_qkv_projections()
        elif hasattr(denoising_network, 'fuse_qkv_projections'):
            denoising_network.fuse_qkv_projections()

    print(f"Model loaded from {args.model}.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    if args.prompt > 0 and args.inference_pipeline == 'samples':
        print(f"Running inference with prompt ...")
        testing_prompts = TESTING_PROMPTS[:args.prompt]
        run_test_inference(
            pipe,
            testing_prompts,
            test_seeds,
            args.device,
            resolution=args.resolution,
            output_path=output_dir,
            deterministic=args.deterministic,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            use_negative_prompts=args.use_negative_prompts,
            subfolder='float')

    # Compute a few-shot calibration set
    few_shot_calibration_prompts = None
    if len(args.few_shot_calibration) > 0:
        args.few_shot_calibration = list(map(int, args.few_shot_calibration))
        pipe.set_progress_bar_config(disable=True)
        few_shot_calibration_prompts = []
        counter = [0]

        def calib_hook(module, inp, inp_kwargs):
            if counter[0] in args.few_shot_calibration:
                few_shot_calibration_prompts.append((inp, inp_kwargs))
            counter[0] += 1
            if counter[0] == args.calibration_steps:
                counter[0] = 0

        h = denoising_network.register_forward_pre_hook(calib_hook, with_kwargs=True)

        run_val_inference(
            pipe,
            args.resolution,
            calibration_prompts,
            test_seeds,
            args.device,
            dtype,
            deterministic=args.deterministic,
            total_steps=args.calibration_steps,
            use_negative_prompts=args.use_negative_prompts,
            test_latents=latents,
            guidance_scale=args.guidance_scale,
            is_unet=is_unet,
            batch=args.calibration_batch_size)
        h.remove()

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    non_blacklist = dict()

    for name, _ in denoising_network.named_modules():
        if any(map(lambda x: x in name, args.quant_recursive_blacklist)):
            blacklist.append(name)
        else:
            if isinstance(_, (torch.nn.Linear, torch.nn.Conv2d)):
                name_to_add = name
                if name_to_add not in non_blacklist:
                    non_blacklist[name_to_add] = 1
                else:
                    non_blacklist[name_to_add] += 1
    print(f"Blacklisted layers: {set(blacklist)}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in denoising_network.modules():
        if hasattr(m, 'lora_layer') and m.lora_layer is not None:
            raise RuntimeError("LoRA layers should be fused in before calling into quantization.")

    def calibration_step(force_full_calibration=False, num_prompts=None):
        if len(args.few_shot_calibration) > 0 and not force_full_calibration:
            for i, (inp_args, inp_kwargs) in enumerate(few_shot_calibration_prompts):
                denoising_network(*inp_args, **inp_kwargs)
                if num_prompts is not None and i == num_prompts:
                    break
        else:
            prompts_subset = calibration_prompts[:num_prompts] if num_prompts is not None else calibration_prompts
            run_val_inference(
                pipe,
                args.resolution,
                prompts_subset,
                test_seeds,
                args.device,
                dtype,
                deterministic=args.deterministic,
                total_steps=args.calibration_steps,
                use_negative_prompts=args.use_negative_prompts,
                test_latents=latents,
                guidance_scale=args.guidance_scale,
                is_unet=is_unet)

    if args.activation_equalization:
        pipe.set_progress_bar_config(disable=True)
        print("Applying Activation Equalization")
        with torch.no_grad(), activation_equalization_mode(
                denoising_network,
                alpha=args.act_eq_alpha,
                layerwise=True,
                blacklist_layers=blacklist if args.exclude_blacklist_act_eq else None,
                add_mul_node=True):
            # Workaround to expose `in_features` attribute from the Hook Wrapper
            for m in denoising_network.modules():
                if isinstance(m, KwargsForwardHook) and hasattr(m.module, 'in_features'):
                    m.in_features = m.module.in_features
            act_eq_num_prompts = 1 if args.dry_run or args.load_checkpoint else len(
                calibration_prompts)

            # SmoothQuant seems to be make better use of all the timesteps
            calibration_step(force_full_calibration=True, num_prompts=act_eq_num_prompts)

        # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
        for m in denoising_network.modules():
            if isinstance(m, EqualizedModule) and hasattr(m.layer, 'in_features'):
                m.in_features = m.layer.in_features

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

    sdpa_kwargs = dict()
    if args.sdpa_scale_stats_op == 'minmax':

        @value
        def sdpa_scale_stats_type():
            if args.sdpa_quant_type == 'asym':
                sdpa_scaling_stats_op = StatsOp.MIN_MAX
            else:
                sdpa_scaling_stats_op = StatsOp.MAX
            return sdpa_scaling_stats_op

        sdpa_kwargs['scaling_stats_op'] = sdpa_scale_stats_type

    if args.sdpa_zp_stats_op == 'minmax':

        @value
        def sdpa_zp_stats_type():
            if args.sdpa_quant_type == 'asym':
                zero_point_stats_impl = NegativeMinOrZero
                return zero_point_stats_impl

        sdpa_kwargs['zero_point_stats_impl'] = sdpa_zp_stats_type

    # Model needs calibration if any of its activation quantizers are 'static'
    activation_bw = [
        args.linear_input_bit_width,
        args.conv_input_bit_width,
        args.sdpa_bit_width,]
    activation_st = [
        args.input_scale_type,
        args.input_scale_type,
        args.sdpa_scale_type,]
    needs_calibration = any(
        map(lambda b, st: (b > 0) and st == 'static', activation_bw, activation_st))

    # Quantize model
    if args.quantize:

        print("Applying model quantization...")
        quantizers = generate_quantizers(
            dtype=dtype,
            device=args.device,
            scale_rounding_func_type=args.scale_rounding_func,
            weight_bit_width=weight_bit_width,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            quantize_input_zero_point=args.quantize_input_zero_point,
            input_group_size=args.input_group_size,
            input_bit_width=input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_type=args.input_scale_type,
            input_scale_precision=args.input_scale_precision,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            input_kwargs=input_kwargs)

        layer_map = generate_quant_maps(
            *quantizers, dtype, args.device, args.input_quant_format, False)

        linear_qkwargs = layer_map[torch.nn.Linear][1]
        linear_qkwargs[
            'input_quant'] = None if args.linear_input_bit_width == 0 else linear_qkwargs[
                'input_quant']
        linear_qkwargs[
            'weight_quant'] = None if args.linear_weight_bit_width == 0 else linear_qkwargs[
                'weight_quant']
        layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], linear_qkwargs)

        conv_qkwargs = layer_map[torch.nn.Conv2d][1]
        conv_qkwargs[
            'input_quant'] = None if args.conv_input_bit_width == 0 else conv_qkwargs['input_quant']
        conv_qkwargs['weight_quant'] = None if args.conv_weight_bit_width == 0 else conv_qkwargs[
            'weight_quant']
        layer_map[torch.nn.Conv2d] = (layer_map[torch.nn.Conv2d][0], conv_qkwargs)

        if args.sdpa_bit_width > 0:
            assert args.share_qkv_quant, "SDPA quantization requires QKV fusion. Enable share_qkv_quant"
            # `args.weight_quant_granularity` must be compatible with `args.sdpa_quant_format`
            sdpa_quantizers = generate_quantizers(
                dtype=dtype,
                device=args.device,
                scale_rounding_func_type=args.scale_rounding_func,
                weight_bit_width=args.sdpa_bit_width,
                weight_quant_format=args.sdpa_quant_format,
                weight_quant_type=args.sdpa_quant_type,
                weight_param_method=args.weight_param_method,
                weight_scale_precision=args.weight_scale_precision,
                weight_quant_granularity=args.weight_quant_granularity,
                weight_group_size=args.weight_group_size,
                quantize_weight_zero_point=args.quantize_weight_zero_point,
                quantize_input_zero_point=args.quantize_sdpa_zero_point,
                input_bit_width=args.sdpa_bit_width,
                input_group_size=args.input_group_size,
                input_quant_format=args.sdpa_quant_format,
                input_scale_type=args.sdpa_scale_type,
                input_scale_precision=args.sdpa_scale_precision,
                input_param_method=args.sdpa_param_method,
                input_quant_type=args.sdpa_quant_type,
                input_quant_granularity=args.sdpa_quant_granularity,
                input_kwargs=sdpa_kwargs)
            # We generate all quantizers, but we are only interested in activation quantization for
            # the output of softmax and the output of QKV
            input_quant = sdpa_quantizers[0]
            extra_kwargs = {
                'fuse_qkv':
                    args.share_qkv_quant,
                'cross_attention_dim':
                    lambda module: module.cross_attention_dim
                    if module.is_cross_attention else None}

            if is_flux:
                extra_kwargs['qk_norm'] = 'rms_norm'
                extra_kwargs['bias'] = True
                extra_kwargs['processor'] = FusedFluxAttnProcessor2_0()
            else:
                warnings.warn("Quantized Attention is supported only for Flux and SDXL")
                extra_kwargs['processor'] = AttnProcessor()

            query_lambda = lambda module: module.query_dim
            rewriter = ModuleToModuleByClass(
                Attention,
                QuantAttention,
                matmul_input_quant=input_quant,
                query_dim=query_lambda,
                dim_head=lambda module: math.ceil(1 / (module.scale ** 2)),
                is_equalized=args.activation_equalization,
                **extra_kwargs)
            import brevitas.config as config
            config.IGNORE_MISSING_KEYS = True
            denoising_network = rewriter.apply(denoising_network)
            config.IGNORE_MISSING_KEYS = False
            denoising_network = denoising_network.to(args.device)
            denoising_network = denoising_network.to(dtype)

            if args.override_conv_quant_config:
                print(
                    f"Overriding Conv2d quantization to weights: {sdpa_quantizers[1]}, inputs: {sdpa_quantizers[2]}"
                )
                conv_qkwargs = layer_map[torch.nn.Conv2d][1]
                conv_qkwargs['input_quant'] = sdpa_quantizers[2]
                conv_qkwargs['weight_quant'] = sdpa_quantizers[1]
                layer_map[torch.nn.Conv2d] = (layer_map[torch.nn.Conv2d][0], conv_qkwargs)

        denoising_network = layerwise_quantize(
            model=denoising_network,
            compute_layer_map=layer_map,
            name_blacklist=blacklist + args.quant_standalone_blacklist)
        denoising_network.eval()
        print("Model quantization applied.")

        pipe.set_progress_bar_config(disable=True)

        with torch.no_grad():
            calibration_step(num_prompts=1)

        if args.load_checkpoint is not None:
            with load_quant_model_mode(denoising_network):
                pipe = pipe.to('cpu')
                print(f"Loading checkpoint: {args.load_checkpoint}... ", end="")
                denoising_network.load_state_dict(
                    torch.load(args.load_checkpoint, map_location='cpu'))
                print(f"Checkpoint loaded!")
            pipe = pipe.to(args.device)

        elif not args.dry_run:
            if needs_calibration:
                print("Applying activation calibration")
                with torch.no_grad(), calibration_mode(denoising_network):
                    calibration_step(force_full_calibration=True)

        if args.svd_quant:
            print("Applying SVDQuant...")
            denoising_network = apply_svd_quant(
                denoising_network,
                blacklist=None,
                rank=args.svd_quant_rank,
                iters=args.svd_quant_iters,
                dtype=torch.float32)
            # Workaround to expose `in_features` attribute from the ErrorCorrectedModule Wrapper
            for m in denoising_network.modules():
                if isinstance(m, ErrorCorrectedModule) and hasattr(m.layer, 'in_features'):
                    m.in_features = m.layer.in_features
            print("SVDQuant applied.")

        if args.compile_ptq:
            for m in denoising_network.modules():
                if hasattr(m, 'compile_quant'):
                    m.compile_quant()
        if args.gptq:
            print("Applying GPTQ. It can take several hours")
            with torch.no_grad(), quant_inference_mode(denoising_network, compile=args.compile_eval, enabled=args.inference_mode):
                with gptq_mode(
                        denoising_network,
                        create_weight_orig=args
                        .bias_correction,  # if we use bias_corr, we need weight_orig
                        use_quant_activations=True,
                        return_forward_output=False,
                        act_order=True) as gptq:
                    for _ in tqdm(range(gptq.num_layers)):
                        calibration_step(num_prompts=128)
                        gptq.update()

        if args.bias_correction:
            print("Applying bias correction")
            with torch.no_grad(), quant_inference_mode(denoising_network, compile=args.compile_eval, enabled=args.inference_mode):
                with bias_correction_mode(denoising_network):
                    calibration_step(force_full_calibration=True)

    if args.vae_fp16_fix and is_sd_xl:
        vae_fix_scale = 128
        layer_whitelist = [
            "decoder.up_blocks.2.upsamplers.0.conv",
            "decoder.up_blocks.3.resnets.0.conv2",
            "decoder.up_blocks.3.resnets.1.conv2",
            "decoder.up_blocks.3.resnets.2.conv2"]

        corrected_layers = []
        with torch.no_grad():
            for name, module in pipe.vae.named_modules():
                if name in layer_whitelist:
                    corrected_layers.append(name)
                    module.weight /= vae_fix_scale
                    if module.bias is not None:
                        module.bias /= vae_fix_scale
        print(f"Corrected layers in VAE: {corrected_layers}")

    if args.vae_quantize:
        assert not is_flux, "Not supported yet"
        print("Quantizing VAE")
        vae_calibration = collect_vae_calibration(
            pipe, calibration_prompts, test_seeds, dtype, latents, args)
        if args.vae_activation_equalization:
            with torch.no_grad(), activation_equalization_mode(
                    pipe.vae,
                    alpha=0.9,#args.vae_act_eq_alpha,
                    layerwise=True,
                    blacklist_layers=blacklist if args.exclude_blacklist_act_eq else None,
                    add_mul_node=True):
                for (inp_args, inp_kwargs) in vae_calibration:
                    input_args = tuple([
                        input_arg.cpu() if isinstance(input_arg, torch.Tensor) else input_arg
                        for input_arg in input_args])
                    input_kwargs = {
                        k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                        for (k, v) in input_kwargs.items()}
                    pipe.vae.decode(*inp_args, **inp_kwargs)

        quantizers = generate_quantizers(
            dtype=dtype,
            device=args.device,
            scale_rounding_func_type=args.scale_rounding_func,
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
            input_group_size=args.input_group_size,
            input_quant_format=args.input_quant_format,
            input_scale_type=args.input_scale_type,
            input_scale_precision=args.input_scale_precision,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            input_kwargs=input_kwargs,
            scaling_min_val=1e-3)

        layer_map = generate_quant_maps(
            *quantizers, dtype, args.device, args.input_quant_format, False)

        linear_qkwargs = layer_map[torch.nn.Linear][1]
        linear_qkwargs[
            'input_quant'] = None if args.linear_input_bit_width == 0 else linear_qkwargs[
                'input_quant']
        linear_qkwargs[
            'weight_quant'] = None if args.linear_weight_bit_width == 0 else linear_qkwargs[
                'weight_quant']
        layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], linear_qkwargs)

        conv_qkwargs = layer_map[torch.nn.Conv2d][1]
        conv_qkwargs[
            'input_quant'] = None if args.conv_input_bit_width == 0 else conv_qkwargs['input_quant']
        conv_qkwargs['weight_quant'] = None if args.conv_weight_bit_width == 0 else conv_qkwargs[
            'weight_quant']
        layer_map[torch.nn.Conv2d] = (layer_map[torch.nn.Conv2d][0], conv_qkwargs)
        pipe.vae = layerwise_quantize(
            model=pipe.vae, compute_layer_map=layer_map, name_blacklist=['conv_out'])

        with torch.no_grad():
            input_args = tuple([
                input_arg.cuda() if isinstance(input_arg, torch.Tensor) else input_arg
                for input_arg in vae_calibration[0][0]])
            input_kwargs = {
                k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                for (k, v) in vae_calibration[0][1].items()}
            pipe.vae.decode(*input_args, **input_kwargs)
        if needs_calibration:
            print("Applying activation calibration")
            with torch.no_grad(), calibration_mode(pipe.vae):
                for (inp_args, inp_kwargs) in vae_calibration:
                    inp_args = tuple([
                        input_arg.cuda() if isinstance(input_arg, torch.Tensor) else input_arg
                        for input_arg in input_args])
                    inp_kwargs = {
                        k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                        for (k, v) in input_kwargs.items()}
                    pipe.vae.decode(*inp_args, **inp_kwargs)

        if args.vae_gptq:
            print("Applying GPTQ")
            with torch.no_grad(), gptq_mode(pipe.vae,
                        create_weight_orig=False,
                        use_quant_activations=False,
                        return_forward_output=True,
                        act_order=True) as gptq:
                for inp_args, inp_kwargs in vae_calibration:
                    inp_args = tuple([
                        input_arg.cuda() if isinstance(input_arg, torch.Tensor) else input_arg
                        for input_arg in input_args])
                    inp_kwargs = {
                        k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                        for (k, v) in input_kwargs.items()}
                    pipe.vae.decode(*inp_args, **inp_kwargs)
        if args.vae_bias_correction:
            print("Applying Bias Correction")
            with torch.no_grad(), bias_correction_mode(pipe.vae):
                for inp_args, inp_kwargs in vae_calibration:
                    inp_args = tuple([
                        input_arg.cuda() if isinstance(input_arg, torch.Tensor) else input_arg
                        for input_arg in input_args])
                    inp_kwargs = {
                        k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                        for (k, v) in input_kwargs.items()}
                    pipe.vae.decode(*inp_args, **inp_kwargs)
        print("VAE quantized")

    if args.checkpoint_name is not None and args.load_checkpoint is None:
        torch.save(denoising_network.state_dict(), os.path.join(output_dir, args.checkpoint_name))
        if args.vae_fp16_fix:
            torch.save(
                pipe.vae.state_dict(), os.path.join(output_dir, f"vae_{args.checkpoint_name}"))

    if args.export_target:
        # Move to cpu and to float32 to enable CPU export
        if args.export_cpu_float32:
            denoising_network.to('cpu').to(torch.float32)
        denoising_network.eval()
        device = next(iter(denoising_network.parameters())).device
        dtype = next(iter(denoising_network.parameters())).dtype

        if args.export_target == 'onnx':
            assert is_sd_xl, "Only SDXL ONNX export is currently supported. If this impacts you, feel free to open an issue"

            trace_inputs = generate_unet_xl_rand_inputs(
                embedding_shape=SD_XL_EMBEDDINGS_SHAPE,
                unet_input_shape=unet_input_shape(args.resolution),
                device=device,
                dtype=dtype)

            if args.weight_quant_granularity == 'per_group':
                export_manager = BlockQuantProxyLevelManager
            else:
                export_manager = StdQCDQONNXManager
                export_manager.change_weight_export(export_weight_q_node=args.export_weight_q_node)
            export_onnx(pipe, trace_inputs, output_dir, export_manager)
        if args.export_target == 'params_only':
            device = next(iter(denoising_network.parameters())).device
            pipe.to('cpu')
            export_quant_params(denoising_network, output_dir, 'denoising_network_')
            if args.vae_quantize or args.vae_fp16_fix:
                export_quant_params(pipe.vae, output_dir, 'vae_')
            else:
                vae_output_path = os.path.join(output_dir, 'vae.safetensors')
                print(f"Saving vae to {vae_output_path} ...")
                save_file(pipe.vae.state_dict(), vae_output_path)
            pipe.to(device)

    # Perform inference
    if args.prompt > 0 and not args.dry_run:
        if args.inference_pipeline == 'mlperf':
            from brevitas_examples.stable_diffusion.mlperf_evaluation.accuracy import \
                compute_mlperf_fid

            print(f"Computing accuracy with MLPerf pipeline")
            with torch.no_grad(), quant_inference_mode(denoising_network, compile=args.compile_eval, enabled=args.inference_mode):
                # Perform a single forward pass before evenutally compiling
                run_val_inference(
                    pipe,
                    args.resolution, [calibration_prompts[0]],
                    test_seeds,
                    args.device,
                    dtype,
                    total_steps=1,
                    deterministic=args.deterministic,
                    use_negative_prompts=args.use_negative_prompts,
                    test_latents=latents,
                    guidance_scale=args.guidance_scale,
                    is_unet=is_unet)
                if args.compile:
                    denoising_network = torch.compile(denoising_network)
                compute_mlperf_fid(
                    args.model,
                    args.path_to_coco,
                    pipe,
                    args.prompt,
                    output_dir,
                    args.device,
                    not args.vae_fp16_fix)
        elif args.inference_pipeline == 'samples':
            print(f"Computing accuracy on default prompt")
            testing_prompts = TESTING_PROMPTS[:args.prompt]
            with torch.no_grad(), quant_inference_mode(denoising_network, compile=args.compile_eval, enabled=args.inference_mode):
                run_val_inference(
                    pipe,
                    args.resolution, [calibration_prompts[0]],
                    test_seeds,
                    args.device,
                    dtype,
                    total_steps=1,
                    deterministic=args.deterministic,
                    use_negative_prompts=args.use_negative_prompts,
                    test_latents=latents,
                    guidance_scale=args.guidance_scale,
                    is_unet=is_unet)

                run_test_inference(
                    pipe,
                    testing_prompts,
                    test_seeds,
                    args.device,
                    resolution=args.resolution,
                    output_path=output_dir,
                    deterministic=args.deterministic,
                    inference_steps=args.inference_steps,
                    use_negative_prompts=args.use_negative_prompts,
                    guidance_scale=args.guidance_scale,
                    subfolder='quant')

            float_images_values = load_images_from_folder(os.path.join(output_dir, 'float'))
            quant_images_values = load_images_from_folder(os.path.join(output_dir, 'quant'))

            fid = FrechetInceptionDistance(normalize=False).to('cuda')
            for float_image in tqdm(float_images_values):
                fid.update(float_image.unsqueeze(0).to('cuda'), real=True)
            for quant_image in tqdm(quant_images_values):
                fid.update(quant_image.unsqueeze(0).to('cuda'), real=False)

            print(f"Torchmetrics FID: {float(fid.compute())}")
            torchmetrics_fid = float(fid.compute())
            # Dump args to json
            with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
                json.dump(vars(args), fp)
            clean_fid = 0.
            if cleanfid is not None:
                score = cleanfid.compute_fid(
                    os.path.join(output_dir, 'float'), os.path.join(output_dir, 'quant'))
                print(f"Cleanfid FID: {float(score)}")
                clean_fid = float(score)
            results = {'torchmetrics_fid': torchmetrics_fid, 'clean_fid': clean_fid}
            with open(os.path.join(output_dir, 'results.json'), 'w') as fp:
                json.dump(results, fp)

        elif args.inference_pipeline == 'reference_images':
            pipe.set_progress_bar_config(disable=True)

            # Load the reference images.
            # We expect a folder with either '.png' or '.jpeg'.
            # All the images will be used for FID computation.
            float_images_values = load_images_from_folder(args.reference_images_path)

            fid = FrechetInceptionDistance(normalize=False).to('cuda')
            for float_image in tqdm(float_images_values):

                if float_image.shape[0] == 1:
                    float_image = float_image.repeat(3, 1, 1)
                if len(float_image.shape) == 3:
                    float_image = float_image.unsqueeze(0)
                fid.update(float_image.cuda(), real=True)
            del float_images_values

            with torch.no_grad(), quant_inference_mode(denoising_network, compile=args.compile_eval, enabled=args.inference_mode):
                run_val_inference(
                    pipe,
                    args.resolution, [calibration_prompts[0]],
                    test_seeds,
                    args.device,
                    dtype,
                    total_steps=1,
                    deterministic=args.deterministic,
                    use_negative_prompts=args.use_negative_prompts,
                    test_latents=latents,
                    guidance_scale=args.guidance_scale,
                    is_unet=is_unet)

                # Load reference test set
                # We expect a pandas dataset with a 'caption' column
                captions_df = pd.read_csv(args.caption_path, sep="\t")
                captions_df = [captions_df.loc[i].caption for i in range(len(captions_df))]
                captions_df = captions_df[:args.prompt]

                run_test_inference(
                    pipe,
                    captions_df,
                    test_seeds,
                    args.device,
                    resolution=args.resolution,
                    output_path=output_dir,
                    deterministic=args.deterministic,
                    inference_steps=args.inference_steps,
                    use_negative_prompts=args.use_negative_prompts,
                    guidance_scale=args.guidance_scale,
                    subfolder='quant_reference')

                quant_images_values = load_images_from_folder(
                    os.path.join(output_dir, 'quant_reference'))
                for quant_image in tqdm(quant_images_values):
                    fid.update(quant_image.unsqueeze(0).to('cuda'), real=False)

                print(f"Torchmetrics FID: {float(fid.compute())}")
                if cleanfid is not None:
                    score = cleanfid.compute_fid(
                        args.reference_images_path, os.path.join(output_dir, 'quant_reference'))
                    print(f"Cleanfid FID: {float(fid.compute())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Diffusion quantization')
    parser.add_argument('-m', '--model', type=str, default=None, help='Path or name of the model.')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0', help='Target device for quantized model.')
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=1,
        help='How many seeds to use for each image during validation. Default: 1')
    parser.add_argument(
        '--prompt', type=int, default=4, help='Number of prompt to use for testing. Default: 4')
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
        'Path to MLPerf compliant Coco dataset. Used when the inference_pipeline is mlperf. Default: None'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        help='Resolution along height and width dimension. Default: 512.')
    parser.add_argument('--svd-quant-rank', type=int, default=32, help='SVDQuant rank. Default: 32')
    parser.add_argument(
        '--svd-quant-iters',
        type=int,
        default=1,
        help='Number of iterations to use for SVDQuant (default: %(default)s).')
    parser.add_argument('--guidance-scale', type=float, default=None, help='Guidance scale.')
    parser.add_argument(
        '--calibration-steps', type=int, default=8, help='Steps used during calibration')
    parser.add_argument(
        '--inference-steps', type=int, default=50, help='Steps used during inference')
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
    add_bool_arg(parser, 'svd-quant', default=False, help='Toggle SVDQuant. Default: Disabled')
    add_bool_arg(
        parser, 'bias-correction', default=False, help='Toggle bias-correction. Default: Disabled')
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
    add_bool_arg(
        parser, 'compile', default=False, help='Compile during inference. Default: Disabled')
    parser.add_argument(
        '--export-target',
        type=str,
        default='',
        choices=['', 'onnx', 'params_only'],
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
        '--conv-input-bit-width',
        type=int,
        default=0,
        help='Input bit width. Default: 0 (not quantized)')
    parser.add_argument(
        '--act-eq-alpha',
        type=float,
        default=0.9,
        help='Alpha for activation equalization. Default: 0.9')
    parser.add_argument(
        '--linear-input-bit-width',
        type=int,
        default=0,
        help='Input bit width. Default: 0 (not quantized).')
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
        default='sym',
        choices=['sym', 'asym'],
        help='Weight quantization type. Default: asym.')
    parser.add_argument(
        '--input-quant-type',
        type=str,
        default='sym',
        choices=['sym', 'asym'],
        help='Input quantization type. Default: asym.')
    parser.add_argument(
        '--weight-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
    parser.add_argument(
        '--input-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Input quantization type. Either int or eXmY, with X+Y==input_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
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
        choices=['per_tensor', 'per_group', 'per_row'],
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
    parser.add_argument(
        '--input-group-size',
        type=int,
        default=16,
        help='Group size for per_group input quantization. Default: 16.')
    parser.add_argument(
        '--sdpa-bit-width',
        type=int,
        default=0,
        help='Scaled dot product attention bit width. Default: 0 (not quantized).')
    parser.add_argument(
        '--sdpa-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help=
        'How scales/zero-point are determined for scaled dot product attention. Default: %(default)s.'
    )
    parser.add_argument(
        '--sdpa-scale-stats-op',
        type=str,
        default='minmax',
        choices=['minmax', 'percentile'],
        help=
        'Define what statistics op to use for scaled dot product attention scale. Default: %(default)s.'
    )
    parser.add_argument(
        '--sdpa-zp-stats-op',
        type=str,
        default='minmax',
        choices=['minmax', 'percentile'],
        help=
        'Define what statistics op to use for scaled dot product attention zero point. Default: %(default)s.'
    )
    parser.add_argument(
        '--sdpa-scale-precision',
        type=str,
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help=
        'Whether the scaled dot product attention scale is a float value or a po2. Default: %(default)s.'
    )
    parser.add_argument(
        '--sdpa-quant-type',
        type=str,
        default='sym',
        choices=['sym', 'asym'],
        help='Scaled dot product attention quantization type. Default: %(default)s.')
    parser.add_argument(
        '--sdpa-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Scaled dot product attention quantization format. Either int or eXmY, with X+Y==input_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: %(default)s.'
    )
    parser.add_argument(
        '--sdpa-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor'],
        help=
        'Granularity for scales/zero-point of scaled dot product attention. Default: %(default)s.')
    parser.add_argument(
        '--sdpa-scale-type',
        type=str,
        default='static',
        choices=['static', 'dynamic'],
        help=
        'Whether to do static or dynamic scaled dot product attention quantization. Default: %(default)s.'
    )
    parser.add_argument(
        '--quant-recursive-blacklist',
        type=str,
        default=[],
        nargs='*',
        metavar='NAME',
        help=
        'A list of module names to exclude from quantization. They are recursively searched in the model architecture. Default: %(default)s'
    )
    parser.add_argument(
        '--quant-standalone-blacklist',
        type=str,
        default=[],
        nargs='*',
        metavar='NAME',
        help='A list of module names to exclude from quantization. Default: %(default)s')
    parser.add_argument(
        '--scale-rounding-func',
        type=str,
        default='floor',
        choices=['floor', 'ceil', 'round'],
        help='Inference pipeline for evaluation.  Default: %(default)s')
    parser.add_argument(
        '--inference-pipeline',
        type=str,
        default='samples',
        choices=['samples', 'reference_images', 'mlperf'],
        help='Inference pipeline for evaluation.  Default: %(default)s')
    parser.add_argument(
        '--caption-path',
        type=str,
        default=None,
        help='Inference pipeline for evaluation.  Default: %(default)s')
    parser.add_argument(
        '--reference-images-path',
        type=str,
        default=None,
        help='Inference pipeline for evaluation.  Default: %(default)s')
    parser.add_argument(
        '--few-shot-calibration',
        default=[],
        nargs='*',
        help='What timesteps to use for few-shot-calibration.  Default: %(default)s')
    parser.add_argument(
        '--calibration-batch-size',
        type=int,
        default=1,
        help='Batch size for few-shot-calibration.  Default: %(default)s')
    add_bool_arg(
        parser,
        'quantize-weight-zero-point',
        default=True,
        help='Quantize weight zero-point. Default: Enabled')
    add_bool_arg(
        parser,
        'exclude-blacklist-act-eq',
        default=False,
        help='Exclude unquantized layers from activation equalization. Default: Disabled')
    add_bool_arg(
        parser,
        'quantize-input-zero-point',
        default=False,
        help='Quantize input zero-point. Default: Enabled')
    add_bool_arg(
        parser,
        'quantize-sdpa-zero-point',
        default=False,
        help='Quantize scaled dot product attention zero-point. Default: %(default)s')
    add_bool_arg(
        parser, 'export-cpu-float32', default=False, help='Export FP32 on CPU. Default: Disabled')
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
        'override-conv-quant-config',
        default=False,
        help='Quantize Convolutions in the same way as SDP (i.e., FP8). Default: Disabled')
    add_bool_arg(
        parser,
        'vae-fp16-fix',
        default=False,
        help='Rescale the VAE to not go NaN with FP16. Default: Disabled')
    add_bool_arg(
        parser,
        'share-qkv-quant',
        default=False,
        help='Share QKV/KV quantization. Default: Disabled')
    add_bool_arg(parser, 'vae-quantize', default=False, help='Quantize VAE. Default: Disabled')
    add_bool_arg(
        parser,
        'vae-activation-equalization',
        default=False,
        help='Activation equalization for VAE, if quantize VAE is Enabled. Default: Disabled')
    add_bool_arg(
        parser,
        'vae-gptq',
        default=False,
        help='GPTQ for VAE, if quantize VAE is Enabled. Default: Disabled')
    add_bool_arg(
        parser,
        'vae-bias-correction',
        default=False,
        help='Bias Correction for VAE, if quantize VAE is Enabled. Default: Disabled')
    add_bool_arg(
        parser, 'compile-ptq', default=False, help='Compile proxies for PTQ. Default: Disabled')
    add_bool_arg(
        parser,
        'compile-eval',
        default=False,
        help='Compile proxies for evaluation. Default: Disabled')
    add_bool_arg(
        parser,
        'inference-mode',
        default=True,
        help='Use inference mode for PTQ and eval. Default: Enabled')
    add_bool_arg(
        parser,
        'deterministic',
        default=True,
        help='Deterministic image generation. Default: Enabled')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
