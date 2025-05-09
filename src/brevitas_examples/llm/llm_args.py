# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from typing import List, Optional
from warnings import warn

from brevitas_examples.common.parse_utils import quant_format_validator


def create_llm_args_parser():
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
        '--dtype',
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help='Data type for model. Default: %(default)s')
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data. Default: 0.')
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help='Number of calibration data samples. Default: 128.')
    parser.add_argument(
        '--nsamples-rot-calibration',
        type=int,
        default=800,
        help='Number of calibration data samples for rotation. Default: %(default)d.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length. Default: 2048.')
    parser.add_argument('--eval', action='store_true', help='Eval model PPL on the chosen Dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wikitext2', 'c4', 'pile'],
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
        '--kv-quant-type',
        type=str,
        default=None,
        choices=['sym', 'asym'],
        help=
        'KV quantization type. If None, it will follow input quant type. If set, will perform only KV cache quantization. Default: None'
    )
    parser.add_argument(
        '--input-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor', 'per_row', 'per_group'],
        help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
    parser.add_argument(
        '--kv-quant-granularity',
        type=str,
        default=None,
        choices=['per_tensor', 'per_row', 'per_group'],
        help=
        'Granularity for scales/zero-point of KV cache. If not set, it will use input-quant-granularity. Default: %(default)s'
    )
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
    parser.add_argument('--magr', action='store_true', help='Apply MagR.')
    parser.add_argument(
        '--magr-alpha', type=float, default=0.01, help='Alpha for MagR. Default: 0.01.')
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
        '--scaling-min-val',
        type=float,
        default=1e-4,
        help='Minimum value to clamp scale to when using bf16 or fp16 quantization.')
    parser.add_argument(
        '--quant-sdpa',
        action='store_true',
        help='Quantize `F.scaled_dot_product_attention` (default: %(default)s)')
    parser.add_argument(
        '--functional-sdpa-quant',
        action='store_true',
        help=
        'Quantize `F.scaled_dot_product_attention` with stateless module and torch_function (default: %(default)s)'
    )
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
        "--optimize-rotations",
        action="store_true",
        default=False,
        help="Whether to optimize the rotations (default: %(default)s).",
    )
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
    parser.add_argument('--svd-quant', action='store_true', help='Apply SVDQuant.')
    parser.add_argument(
        '--svd-quant-rank',
        type=int,
        default=32,
        help='Rank to use for SVDQuant (default: %(default)s).')
    parser.add_argument(
        '--svd-quant-iters',
        type=int,
        default=1,
        help='Number of iterations to use for SVDQuant (default: %(default)s).')
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
        '--load-checkpoint',
        action="store_true",
        help='Boolean flag to load_checkpoint, uses checkpoint_name. Default %(default)s)')
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
        type=str,
        default=None,
        choices=['lm_eval', 'lighteval'],
        help='Perform zero_shot evaluation with lm_eval or lighteval. Default %(default)s)')
    parser.add_argument('--few-shot-override-batch-size', type=int, default=None)
    parser.add_argument(
        '--compile-ptq',
        default=False,
        action="store_true",
        help='Compile for PTQ algorithms. Default %(default)s)')
    parser.add_argument(
        '--compile-eval',
        default=False,
        action="store_true",
        help='Compile during evaluation. Default %(default)s)')
    parser.add_argument(
        '--few-shot-zeroshot',
        default=False,
        action="store_true",
        help='Whether to do zero or few shot eval. Default %(default)s)')
    parser.add_argument(
        '--bos-preprocessing',
        type=str,
        default=None,
        choices=[None, 'document', 'sequence'],
        help=
        'Type of BOS token pre-processing for training and evaluation datasets. Default %(default)s)'
    )
    parser.add_argument(
        '--few-shot-limit', type=int, default=None, help='Few shot limit. Default %(default)s)')
    parser.add_argument(
        '--few-shot-tasks',
        default=['arc_challenge', 'arc_easy', 'winogrande', 'piqa'],
        type=str,
        nargs='*',
        help='A list of tasks for zero_shot evaluation. Default: %(default)s')
    parser.add_argument(
        '--rotation-layers-to-expand',
        type=str,
        default=[],
        nargs='*',
        help='A list of module names to expand with hadamard rotation. Default: %(default)s')
    parser.add_argument(
        "--awq-scale",
        action="store_true",
        help="Whether to apply AWQ scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--awq-clip",
        action="store_true",
        help="Whether to apply AWQ clipping (default: %(default)s).",
    )
    return parser


def validate(args, extra_args: Optional[List[str]] = None):
    if args.optimize_rotations:
        assert args.rotation in ['fx', 'fused_no_fx'], f"Rotations can only be optimized if --rotation=fx or --rotation=fused_no_fx"
    else:
        assert extra_args is None or len(extra_args) == 0, f"The following unknown arguments were passed: {[extra_arg for extra_arg in extra_args if extra_arg.startswith('--')]}"
    if args.functional_sdpa_quant:
        assert args.input_scale_type == 'dynamic' or args.input_bit_width is None, "Functional SDPA Quant requires dynamic activation quantization"
    if args.rotation == 'fx':
        assert args.ln_affine_merge, 'Graph rotation requires to merge LN/RMS norm affine parameters'
        assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
        assert args.convert_layernorm_to_rmsnorm, 'Graph rotation requires to replace LayerNorm with RMSNorm'
    elif args.rotation == 'fused_no_fx':
        assert not args.convert_layernorm_to_rmsnorm, 'LayerNorm is automatically replaced with RMSNorm when running with --rotation=fused_no_fx. Remove the flag --convert-layernorm-to-rmsnorm'
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
