"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from datetime import datetime
import json
import os
import time

from dependencies import value
from diffusers import DiffusionPipeline
import torch
from torch import nn
from tqdm import tqdm

from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.export.torch.qcdq.manager import TorchQCDQManager
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.utils.torch_utils import KwargsForwardHook
from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_2_1_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_XL_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.export import export_onnx
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_latents
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_21_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_xl_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import unet_input_shape

TEST_SEED = 123456

VALIDATION_PROMPTS = {
    'validation_prompt_0': 'A cat playing with a ball',
    'validation_prompt_1': 'A dog running on the beach'}


def run_test_inference(
        pipe, resolution, prompts, seeds, output_path, device, dtype, name_prefix=''):
    with torch.no_grad():
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        test_latents = generate_latents(seeds, device, dtype, unet_input_shape(resolution))

        for name, prompt in prompts.items():
            print(f"Generating: {name}")
            images = pipe([prompt] * len(seeds), latents=test_latents).images
            for i, seed in enumerate(seeds):
                file_path = os.path.join(output_path, f"{name_prefix}{name}_{seed}.png")
                print(f"Saving to {file_path}")
                images[i].save(file_path)


def run_val_inference(pipe, resolution, prompts, seeds, output_path, device, dtype, name_prefix=''):
    with torch.no_grad():
        test_latents = generate_latents(seeds, device, dtype, unet_input_shape(resolution))

        for name, prompt in prompts.items():
            print(f"Generating: {name}")
            # We don't want to generate any image, so we return only the latent encoding pre VAE
            pipe([prompt] * len(seeds), latents=test_latents, output_type='latent')


def main(args):

    if args.export_target:
        assert args.weight_quant_format == 'int', "Currently only integer quantization supported for export."

    # Select dtype
    if args.float16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create output dir. Move to tmp if None
    ts = datetime.fromtimestamp(time.time())
    str_ts = ts.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f'{str_ts}')
    os.mkdir(output_dir)

    # Dump args to json
    with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # Extend seeds based on batch_size
    test_seeds = [TEST_SEED] + [TEST_SEED + i for i in range(1, args.batch_size)]

    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    print(f"Model loaded from {args.model}.")

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    for name, _ in pipe.unet.named_modules():
        if 'time_emb' in name or 'conv_in' in name:
            blacklist.append(name)
    print(f"Blacklisted layers: {blacklist}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in pipe.unet.modules():
        if hasattr(m, 'lora_layer') and m.lora_layer is not None:
            raise RuntimeError("LoRA layers should be fused in before calling into quantization.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    if args.activation_equalization:
        with activation_equalization_mode(pipe.unet, alpha=0.5, layerwise=True, add_mul_node=True):
            # Workaround to expose `in_features` attribute from the Hook Wrapper
            for m in pipe.unet.modules():
                if isinstance(m, KwargsForwardHook) and hasattr(m.module, 'in_features'):
                    m.in_features = m.module.in_features
            prompts = VALIDATION_PROMPTS
            run_val_inference(
                pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)

        # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
        for m in pipe.unet.modules():
            if isinstance(m, EqualizedModule) and hasattr(m.module, 'in_features'):
                m.in_features = m.module.in_features

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

        # XOR between the two input_bit_width. Either they are both None, or neither of them is
        assert (args.linear_input_bit_width is None) == (args.conv_input_bit_width is None), 'Both input bit width must be specified or left to None'

        is_input_quantized = args.linear_input_bit_width is not None and args.conv_input_bit_width is not None
        if is_input_quantized:

            @value
            def input_bit_width(module):
                if isinstance(module, nn.Linear):
                    return args.linear_input_bit_width
                elif isinstance(module, nn.Conv2d):
                    return args.conv_input_bit_width
                else:
                    raise RuntimeError(f"Module {module} not supported.")
        else:
            input_bit_width = None

        print("Applying model quantization...")
        quantize_model(
            pipe.unet,
            dtype=dtype,
            device=args.device,
            name_blacklist=blacklist,
            weight_bit_width=weight_bit_width,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            input_bit_width=input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_type=args.input_scale_type,
            input_scale_precision=args.input_scale_precision,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity)
        print("Model quantization applied.")

        if is_input_quantized and args.input_scale_type == 'static':
            print("Applying activation calibration")
            with calibration_mode(pipe.unet):
                prompts = VALIDATION_PROMPTS
                run_val_inference(
                    pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)

        if args.gptq:
            print("Applying GPTQ. It can take several hours")
            with gptq_mode(pipe.unet,
                           create_weight_orig=False,
                           use_quant_activations=False,
                           return_forward_output=True,
                           act_order=True) as gptq:
                prompts = VALIDATION_PROMPTS
                for _ in tqdm(range(gptq.num_layers)):
                    run_val_inference(
                        pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)
                    gptq.update()

        print("Applying bias correction")
        with bias_correction_mode(pipe.unet):
            prompts = VALIDATION_PROMPTS
            run_val_inference(
                pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)

    # Perform inference
    if args.prompt:
        print(f"Running inference with prompt '{args.prompt}' ...")
        prompts = {'manual_prompt': args.prompt}
        run_test_inference(
            pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)

    if args.export_target:
        # Move to cpu and to float32 to enable CPU export
        if not (args.float16 and args.export_cuda_float16):
            pipe.unet.to('cpu').to(torch.float32)
        pipe.unet.eval()
        device = next(iter(pipe.unet.parameters())).device
        dtype = next(iter(pipe.unet.parameters())).dtype

        # Define tracing input
        if args.is_sd_xl:
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
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size. Default: 4')
    parser.add_argument(
        '--prompt',
        type=str,
        default='An austronaut riding a horse on Mars.',
        help='Manual prompt for testing. Default: An austronaut riding a horse on Mars.')
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Resolution along height and width dimension. Default: 512.')
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
    add_bool_arg(parser, 'float16', default=True, help='Enable float16 execution. Default: Enabled')
    add_bool_arg(
        parser,
        'attention-slicing',
        default=False,
        help='Enable attention slicing. Default: Disabled')
    add_bool_arg(
        parser,
        'is-sd-xl',
        default=False,
        help='Enable this flag to correctly export SDXL. Default: Disabled')
    parser.add_argument(
        '--export-target', type=str, default='', choices=['', 'onnx'], help='Target export flow.')
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
        default=None,
        help='Input bit width. Default: None (not quantized)')
    parser.add_argument(
        '--linear-input-bit-width',
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
        parser, 'export-cuda-float16', default=False, help='Export FP16 on CUDA. Default: Disabled')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
