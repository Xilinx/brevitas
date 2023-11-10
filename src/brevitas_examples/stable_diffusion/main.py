"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from datetime import datetime
import json
import os
import re
import time

from diffusers import StableDiffusionPipeline
import torch
from torch import nn

from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.stable_diffusion.sd_quant.constants import SD_2_1_EMBEDDINGS_SHAPE
from brevitas_examples.stable_diffusion.sd_quant.export import export_torchscript_weight_group_quant
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_latents
from brevitas_examples.stable_diffusion.sd_quant.utils import generate_unet_rand_inputs
from brevitas_examples.stable_diffusion.sd_quant.utils import unet_input_shape

TEST_SEED = 123456


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


def main(args):

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
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
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

    # Quantize model
    if args.quantize:

        def bit_width_fn(module):
            if isinstance(module, nn.Linear):
                return args.linear_weight_bit_width
            elif isinstance(module, nn.Conv2d):
                return args.conv_weight_bit_width
            else:
                raise RuntimeError(f"Module {module} not supported.")

        weight_bit_width = lambda module: bit_width_fn(module)

        print("Applying model quantization...")
        quantize_model(
            pipe.unet,
            dtype=dtype,
            name_blacklist=blacklist,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_bit_width=weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point)
        print("Model quantization applied.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    # Perform inference
    if args.prompt:
        print(f"Running inference with prompt '{args.prompt}' ...")
        prompts = {'manual_prompt': args.prompt}
        run_test_inference(
            pipe, args.resolution, prompts, test_seeds, output_dir, args.device, dtype)

    if args.export_target:
        # Move to cpu and to float32 to enable CPU export
        pipe.unet.to('cpu').to(torch.float32)
        pipe.unet.eval()
    if args.export_target == 'torchscript_weight_group_quant':
        assert args.weight_quant_granularity == 'per_group', "Per-group quantization required."
        assert args.weight_quant_format == 'int', "Only integer quantization supported for export."
        trace_inputs = generate_unet_rand_inputs(
            embedding_shape=SD_2_1_EMBEDDINGS_SHAPE,
            unet_input_shape=unet_input_shape(args.resolution),
            device='cpu',
            dtype=torch.float32)
        export_torchscript_weight_group_quant(pipe, trace_inputs, output_dir)
    else:
        raise ValueError(f"{args.export_target} not recognized.")


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
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size.')
    parser.add_argument(
        '--prompt',
        type=str,
        default='An austronaut riding a horse on Mars.',
        help='Manual prompt for testing.')
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
    add_bool_arg(parser, 'quantize', default=True, help='Toggle quantization.')
    add_bool_arg(parser, 'float16', default=True, help='Enable float16 execution.')
    add_bool_arg(parser, 'attention-slicing', default=False, help='Enable attention slicing.')
    parser.add_argument(
        '--export-target',
        type=str,
        default='',
        choices=['', 'torchscript_weight_group_quant'],
        help='Target export flow.')
    parser.add_argument(
        '--conv-weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
    parser.add_argument(
        '--linear-weight-bit-width', type=int, default=8, help='Weight bit width. Default: 4.')
    parser.add_argument(
        '--weight-param-method',
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
        '--weight-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Weight quantization type. Default: asym.')
    parser.add_argument(
        '--weight-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. Default: int.')
    parser.add_argument(
        '--weight-quant-granularity',
        type=str,
        default='per_group',
        choices=['per_channel', 'per_tensor', 'per_group'],
        help='Granularity for scales/zero-point of weights. Default: per_group.')
    parser.add_argument(
        '--weight-group-size',
        type=int,
        default=16,
        help='Group size for per_group weight quantization. Default: 16.')
    add_bool_arg(
        parser, 'quantize-weight-zero-point', default=True, help='Quantize weight zero-point.')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
