# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from functools import partial
from typing import List, Optional

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import create_entrypoint_args_parser
from brevitas_examples.common.parse_utils import parse_type

try:
    import torchvision
except ImportError:
    torchvision = None


def create_args_parser() -> ArgumentParser:
    parser = create_entrypoint_args_parser(description='PyTorch ImageNet PTQ Validation')
    parser.add_argument(
        '--calibration-dir',
        type=str,
        default=None,
        help='Path to folder containing Imagenet calibration folder')
    parser.add_argument(
        '--validation-dir',
        type=str,
        default=None,
        help='Path to folder containing Imagenet validation folder')
    parser.add_argument(
        '--workers', default=8, type=int, help='Number of data loading workers (default: 8)')
    parser.add_argument(
        '--batch-size-calibration',
        default=64,
        type=int,
        help='Minibatch size for calibration (default: 64)')
    parser.add_argument(
        '--batch-size-validation',
        default=256,
        type=int,
        help='Minibatch size for validation (default: 256)')
    parser.add_argument(
        '--export-dir', default='.', type=str, help='Directory where to store the exported models')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use (default: None)')
    parser.add_argument(
        '--calibration-samples', default=1000, type=int, help='Calibration size (default: 1000)')
    # Set choices to model_names if torchvision is available
    if torchvision is not None:
        model_names = sorted(
            name for name in torchvision.models.__dict__
            if name.islower() and not name.startswith("__") and
            callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))
        # Set choices to model_names if torchvision is available
        parser.add_argument(
            '--model-name',
            default='resnet18',
            metavar='ARCH',
            choices=model_names,
            help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    else:
        parser.add_argument(
            '--model-name',
            default='resnet18',
            metavar='ARCH',
            help='model architecture (default: resnet18)')
    parser.add_argument(
        '--dtype',
        default='float',
        choices=['float', 'bfloat16', 'float16'],
        help='Data type to use (float for FP32, bfloat16 for BF16, or float16 for FP16)')
    parser.add_argument(
        '--target-backend',
        default='fx',
        choices=['fx', 'layerwise', 'flexml'],
        help='Backend to target for quantization (default: fx)')
    parser.add_argument(
        '--scale-factor-type',
        default='float_scale',
        choices=['float_scale', 'po2_scale'],
        help='Type for scale factors (default: float_scale)')
    parser.add_argument(
        '--act-bit-width', default=8, type=int, help='Activations bit width (default: 8)')
    parser.add_argument(
        '--weight-bit-width', default=8, type=int, help='Weights bit width (default: 8)')
    parser.add_argument(
        '--layerwise-first-last-bit-width',
        default=8,
        type=int,
        help='Input and weights bit width for first and last layer w/ layerwise backend (default: 8)'
    )
    parser.add_argument(
        '--bias-bit-width',
        default=32,
        type=partial(parse_type, default_type=int),
        choices=[32, 16, None],
        help='Bias bit width (default: 32)')
    parser.add_argument(
        '--act-quant-type',
        default='sym',
        choices=['sym', 'asym'],
        help='Activation quantization type (default: sym)')
    parser.add_argument(
        '--weight-quant-type',
        default='sym',
        choices=['sym', 'asym'],
        help='Weight quantization type (default: sym)')
    parser.add_argument(
        '--weight-quant-granularity',
        default='per_tensor',
        choices=['per_tensor', 'per_channel', 'per_group'],
        help='Weight quantization type (default: per_tensor)')
    parser.add_argument(
        '--act-quant-granularity',
        default='per_tensor',
        choices=['per_tensor', 'per_group'],
        help='Activation quantization type (default: per_tensor)')
    parser.add_argument(
        '--weight-quant-calibration-type',
        default='stats',
        choices=['stats', 'mse', 'hqo'],
        help='Weight quantization calibration type (default: stats)')
    parser.add_argument(
        '--act-equalization',
        default=None,
        choices=['fx', 'layerwise', None],
        help='Activation equalization type (default: None)')
    parser.add_argument(
        '--act-quant-calibration-type',
        default='stats',
        choices=['stats', 'mse'],
        help='Activation quantization calibration type (default: stats)')
    parser.add_argument(
        '--act-scale-computation-type',
        default='static',
        choices=['static', 'dynamic'],
        help='Activation quantization scale computation type (default: static)')
    parser.add_argument(
        '--graph-eq-iterations',
        default=20,
        type=int,
        help='Numbers of iterations for graph equalization (default: 20)')
    parser.add_argument(
        '--learned-round',
        default=None,
        type=str,
        choices=[None, 'linear_round', 'hard_sigmoid_round', 'sigmoid_round'],
        help='Learned round type (default: None)')
    parser.add_argument(
        '--learned-round-block-name',
        type=str,
        default="layer\d+",
        help='Block name for learned round. It works only if FX is not needed (default: %(default)s)'
    )
    parser.add_argument(
        '--learned-round-loss',
        default='regularised_mse',
        type=str,
        choices=['regularised_mse', 'mse'],
        help='Learned round type (default: none)')
    parser.add_argument(
        '--learned-round-mode',
        default='layerwise',
        choices=['layerwise', 'blockwise'],
        help='Learned round mode (default: none)')
    parser.add_argument(
        '--learned-round-iters',
        default=1000,
        type=int,
        help='Numbers of iterations for learned round for each layer (default: 1000)')
    parser.add_argument(
        '--learned-round-lr-scheduler',
        default=None,
        type=str,
        choices=[None, 'linear'],
        help='Learning rate scheduler for learned round (default: None)')
    parser.add_argument(
        '--learned-round-lr',
        default=1e-3,
        type=float,
        help='Learning rate for learned round (default: 1e-3)')
    parser.add_argument(
        '--learned-round-batch-size',
        default=1,
        type=int,
        help='Learning rate for learned round (default: %(default)d)')
    parser.add_argument(
        '--act-quant-percentile',
        default=99.999,
        type=float,
        help='Percentile to use for stats of activation quantization (default: 99.999)')
    parser.add_argument(
        '--export-onnx-qcdq',
        action='store_true',
        help='If true, export the model in onnx qcdq format')
    parser.add_argument(
        '--export-torch-qcdq',
        action='store_true',
        help='If true, export the model in torch qcdq format')

    add_bool_arg(
        parser,
        'bias-corr',
        default=True,
        help='Bias correction after calibration (default: enabled)')
    add_bool_arg(
        parser,
        'graph-eq-merge-bias',
        default=True,
        help='Merge bias when performing graph equalization (default: enabled)')
    add_bool_arg(
        parser,
        'weight-narrow-range',
        default=False,
        help='Narrow range for weight quantization (default: disabled)')
    parser.add_argument(
        '--gpfq-p', default=1.0, type=float, help='P parameter for GPFQ (default: 1.0)')
    parser.add_argument(
        '--quant-format',
        default='int',
        choices=['int', 'float', 'float_ocp'],
        help='Quantization format to use for weights and activations (default: int)')
    parser.add_argument(
        '--layerwise-first-last-mantissa-bit-width',
        default=4,
        type=int,
        help=
        'Mantissa bit width used with float layerwise quantization for first and last layer (default: 4)'
    )
    parser.add_argument(
        '--layerwise-first-last-exponent-bit-width',
        default=3,
        type=int,
        help=
        'Exponent bit width used with float layerwise quantization for first and last layer (default: 3)'
    )
    parser.add_argument(
        '--weight-mantissa-bit-width',
        default=4,
        type=int,
        help='Mantissa bit width used with float quantization for weights (default: 4)')
    parser.add_argument(
        '--weight-exponent-bit-width',
        default=3,
        type=int,
        help='Exponent bit width used with float quantization for weights (default: 3)')
    parser.add_argument(
        '--act-mantissa-bit-width',
        default=4,
        type=int,
        help='Mantissa bit width used with float quantization for activations (default: 4)')
    parser.add_argument(
        '--act-exponent-bit-width',
        default=3,
        type=int,
        help='Exponent bit width used with float quantization for activations (default: 3)')
    parser.add_argument(
        '--gpxq-accumulator-bit-width',
        default=None,
        type=int,
        help='Accumulator Bit Width for GPxQ (default: None)')
    parser.add_argument(
        '--gpxq-accumulator-tile-size',
        default=None,
        type=int,
        help='Accumulator tile size for GPxQ (default: None)')
    parser.add_argument('--onnx-opset-version', default=None, type=int, help='ONNX opset version')
    parser.add_argument(
        '--channel-splitting-ratio',
        default=0.0,
        type=float,
        help=
        'Split Ratio for Channel Splitting. When set to 0.0, Channel Splitting will not be applied. (default: 0.0)'
    )
    parser.add_argument(
        '--optimizer',
        default='adam',
        choices=['adam', 'sign_sgd'],
        help='Optimizer to use with learnable rounding (default: %(default)s)')
    add_bool_arg(parser, 'gptq', default=False, help='GPTQ (default: disabled)')
    add_bool_arg(parser, 'gpfq', default=False, help='GPFQ (default: disabled)')
    add_bool_arg(parser, 'qronos', default=False, help='Qronos (default: disabled)')
    add_bool_arg(
        parser,
        'gpxq-act-order',
        default=False,
        help='GPxQ Act order heuristic (default: disabled)')
    add_bool_arg(
        parser,
        'gptq-use-quant-activations',
        default=False,
        help='Use quant activations for GPTQ (default: disabled)')
    add_bool_arg(
    parser,
    'disable-create-weight-orig',
    default=False,
    help='Disable maintaining original weights for non-quant forward pass (default: enabled)')
    add_bool_arg(parser, 'calibrate-bn', default=False, help='Calibrate BN (default: disabled)')
    add_bool_arg(
        parser,
        'channel-splitting-split-input',
        default=False,
        help='Input Channels Splitting for channel splitting (default: disabled)')
    add_bool_arg(
        parser,
        'merge-bn',
        default=True,
        help='Merge BN layers before quantizing the model (default: enabled)')
    add_bool_arg(
        parser,
        'uint-sym-act-for-unsigned-values',
        default=True,
        help='Use unsigned act quant when possible (default: enabled)')
    add_bool_arg(parser, 'compile', default=False, help='Use torch.compile (default: disabled)')
    return parser


def validate(args: Namespace, extra_args: Optional[List[str]] = None):
    assert args.calibration_dir is not None, "Specify in --calibration-dir the path to the folder containing the Imagenet calibration dataset."
    assert args.validation_dir is not None, "Specify in --calibration-dir the path to the folder containing the Imagenet validation dataset."
    if args.learned_round:
        assert args.target_backend == "layerwise", "Currently, learned round is only supported with target-backend=layerwise"
