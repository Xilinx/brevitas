# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from typing import List, Optional

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import create_entrypoint_args_parser
from brevitas_examples.common.parse_utils import quant_format_validator


def create_args_parser():
    parser = create_entrypoint_args_parser(description="Stable Diffusion quantization")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='Path or name of the model.')
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
    parser.add_argument('--guidance-scale', type=float, default=7.0, help='Guidance scale.')
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
    return parser


def validate(args: Namespace, extra_args: Optional[List[str]] = None):
    pass
