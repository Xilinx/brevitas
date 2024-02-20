# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from functools import partial
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_act_equalization
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gpfq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_learned_round_learning
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate_bn
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.ptq.utils import add_bool_arg
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")


def parse_type(v, default_type):
    if v == 'None':
        return None
    else:
        return default_type(v)


model_names = sorted(
    name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
    callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))

parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument(
    '--calibration-dir',
    required=True,
    help='Path to folder containing Imagenet calibration folder')
parser.add_argument(
    '--validation-dir', required=True, help='Path to folder containing Imagenet validation folder')
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
parser.add_argument(
    '--model-name',
    default='resnet18',
    metavar='ARCH',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument(
    '--dtype', default='float', choices=['float', 'bfloat16'], help='Data type to use')
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
    help='Input and weights bit width for first and last layer w/ layerwise backend (default: 8)')
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
    choices=['per_tensor', 'per_channel'],
    help='Activation quantization type (default: per_tensor)')
parser.add_argument(
    '--weight-quant-calibration-type',
    default='stats',
    choices=['stats', 'mse'],
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
    '--graph-eq-iterations',
    default=20,
    type=int,
    help='Numbers of iterations for graph equalization (default: 20)')
parser.add_argument(
    '--learned-round-iters',
    default=1000,
    type=int,
    help='Numbers of iterations for learned round for each layer (default: 1000)')
parser.add_argument(
    '--learned-round-lr',
    default=1e-3,
    type=float,
    help='Learning rate for learned round (default: 1e-3)')
parser.add_argument(
    '--act-quant-percentile',
    default=99.999,
    type=float,
    help='Percentile to use for stats of activation quantization (default: 99.999)')
parser.add_argument(
    '--export-onnx-qcdq', action='store_true', help='If true, export the model in onnx qcdq format')
parser.add_argument(
    '--export-torch-qcdq',
    action='store_true',
    help='If true, export the model in torch qcdq format')
add_bool_arg(
    parser,
    'scaling-per-output-channel',
    default=True,
    help='Weight scaling per output channel (default: enabled)')
add_bool_arg(
    parser, 'bias-corr', default=True, help='Bias correction after calibration (default: enabled)')
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
parser.add_argument('--gpfq-p', default=1.0, type=float, help='P parameter for GPFQ (default: 1.0)')
parser.add_argument(
    '--quant-format',
    default='int',
    choices=['int', 'float'],
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
    '--accumulator-bit-width',
    default=None,
    type=int,
    help='Accumulator Bit Width for GPFA2Q (default: None)')
parser.add_argument('--onnx-opset-version', default=None, type=int, help='ONNX opset version')
parser.add_argument(
    '--channel-splitting-ratio',
    default=0.0,
    type=float,
    help=
    'Split Ratio for Channel Splitting. When set to 0.0, Channel Splitting will not be applied. (default: 0.0)'
)
add_bool_arg(parser, 'gptq', default=False, help='GPTQ (default: disabled)')
add_bool_arg(parser, 'gpfq', default=False, help='GPFQ (default: disabled)')
add_bool_arg(parser, 'gpfa2q', default=False, help='GPFA2Q (default: disabled)')
add_bool_arg(
    parser, 'gpxq-act-order', default=False, help='GPxQ Act order heuristic (default: disabled)')
add_bool_arg(parser, 'learned-round', default=False, help='Learned round (default: disabled)')
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


def main():
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.act_quant_calibration_type == 'stats':
        act_quant_calib_config = str(args.act_quant_percentile) + 'stats'
    else:
        act_quant_calib_config = args.act_quant_calibration_type

    if args.act_bit_width == 0:
        args.act_bit_width = None

    config = (
        f"{args.model_name}_"
        f"{args.target_backend}_"
        f"{args.quant_format}_"
        f"{str(args.weight_mantissa_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.weight_exponent_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.act_mantissa_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{str(args.act_exponent_bit_width) + '_' if args.quant_format == 'float' else ''}"
        f"{args.scale_factor_type}_"
        f"a{args.act_bit_width}"
        f"w{args.weight_bit_width}_"
        f"{'gptq_' if args.gptq else ''}"
        f"{'gpfq_' if args.gpfq else ''}"
        f"{'gpfa2q_' if args.gpfa2q else ''}"
        f"{'gpxq_act_order_' if args.gpxq_act_order else ''}"
        f"{'learned_round_' if args.learned_round else ''}"
        f"{'weight_narrow_range_' if args.weight_narrow_range else ''}"
        f"{args.bias_bit_width}bias_"
        f"{args.weight_quant_granularity}_"
        f"{args.act_quant_type}_"
        f"{'bc_' if args.bias_corr else ''}"
        f"{args.graph_eq_iterations}geiters_"
        f"{'mb_' if args.graph_eq_merge_bias else ''}"
        f"{act_quant_calib_config}_"
        f"{args.weight_quant_calibration_type}_"
        f"{'bnc_' if args.calibrate_bn else ''}"
        f"{'channel_splitting' if args.channel_splitting_ratio else ''}")

    print(
        f"Model: {args.model_name} - "
        f"Target backend: {args.target_backend} - "
        f"Quantization type: {args.scale_factor_type} - "
        f"Activation bit width: {args.act_bit_width} - "
        f"Weight bit width: {args.weight_bit_width} - "
        f"GPTQ: {args.gptq} - "
        f"GPFQ: {args.gpfq} - "
        f"GPFA2Q: {args.gpfa2q} - "
        f"GPFQ P: {args.gpfq_p} - "
        f"GPxQ Act Order: {args.gpxq_act_order} - "
        f"GPFA2Q Accumulator Bit Width: {args.accumulator_bit_width} - "
        f"Learned Round: {args.learned_round} - "
        f"Weight narrow range: {args.weight_narrow_range} - "
        f"Bias bit width: {args.bias_bit_width} - "
        f"Weight scale factors type: {args.weight_quant_granularity} - "
        f"Activation quant type: {args.act_quant_type} - "
        f"Bias Correction Enabled: {args.bias_corr} - "
        f"Iterations for graph equalization: {args.graph_eq_iterations} - "
        f"Merge bias in graph equalization: {args.graph_eq_merge_bias} - "
        f"Activation quant calibration type: {act_quant_calib_config} - "
        f"Weight quant calibration type: {args.weight_quant_calibration_type} - "
        f"Calibrate BN: {args.calibrate_bn} - "
        f"Channel Splitting Ratio: {args.channel_splitting_ratio} - "
        f"Split Input: {args.channel_splitting_split_input} - "
        f"Merge BN: {args.merge_bn}")

    # Get model-specific configurations about input shapes and normalization
    model_config = get_model_config(args.model_name)

    # Generate calibration and validation dataloaders
    resize_shape = model_config['resize_shape']
    center_crop_shape = model_config['center_crop_shape']
    inception_preprocessing = model_config['inception_preprocessing']
    calib_loader = generate_dataloader(
        args.calibration_dir,
        args.batch_size_calibration,
        args.workers,
        resize_shape,
        center_crop_shape,
        args.calibration_samples,
        inception_preprocessing)
    val_loader = generate_dataloader(
        args.validation_dir,
        args.batch_size_validation,
        args.workers,
        resize_shape,
        center_crop_shape,
        inception_preprocessing=inception_preprocessing)

    # Get the model from torchvision
    model = get_torchvision_model(args.model_name)
    model = model.to(dtype)

    # Preprocess the model for quantization
    if args.target_backend == 'flexml':
        # flexml requires static shapes, pass a representative input in
        img_shape = model_config['center_crop_shape']
        model = preprocess_for_flexml_quantize(
            model,
            torch.ones(1, 3, img_shape, img_shape, dtype=dtype),
            equalize_iters=args.graph_eq_iterations,
            equalize_merge_bias=args.graph_eq_merge_bias,
            merge_bn=not args.calibrate_bn)
    elif args.target_backend == 'fx' or args.target_backend == 'layerwise':
        model = preprocess_for_quantize(
            model,
            equalize_iters=args.graph_eq_iterations,
            equalize_merge_bias=args.graph_eq_merge_bias,
            merge_bn=args.merge_bn,
            channel_splitting_ratio=args.channel_splitting_ratio,
            channel_splitting_split_input=args.channel_splitting_split_input)
    else:
        raise RuntimeError(f"{args.target_backend} backend not supported.")

    # If available, use the selected GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = False

    if args.act_equalization is not None:
        print("Applying activation equalization:")
        apply_act_equalization(model, calib_loader, layerwise=args.act_equalization == 'layerwise')
    device = next(iter(model.parameters())).device
    # Define the quantized model
    quant_model = quantize_model(
        model,
        dtype=dtype,
        device=device,
        backend=args.target_backend,
        scale_factor_type=args.scale_factor_type,
        bias_bit_width=args.bias_bit_width,
        weight_bit_width=args.weight_bit_width,
        weight_narrow_range=args.weight_narrow_range,
        weight_param_method=args.weight_quant_calibration_type,
        weight_quant_granularity=args.weight_quant_granularity,
        weight_quant_type=args.weight_quant_type,
        layerwise_first_last_bit_width=args.layerwise_first_last_bit_width,
        act_bit_width=args.act_bit_width,
        act_param_method=args.act_quant_calibration_type,
        act_quant_percentile=args.act_quant_percentile,
        act_quant_type=args.act_quant_type,
        quant_format=args.quant_format,
        layerwise_first_last_mantissa_bit_width=args.layerwise_first_last_mantissa_bit_width,
        layerwise_first_last_exponent_bit_width=args.layerwise_first_last_exponent_bit_width,
        weight_mantissa_bit_width=args.weight_mantissa_bit_width,
        weight_exponent_bit_width=args.weight_exponent_bit_width,
        act_mantissa_bit_width=args.act_mantissa_bit_width,
        act_exponent_bit_width=args.act_exponent_bit_width)

    # Calibrate the quant_model on the calibration dataloader
    print("Starting activation calibration:")
    calibrate(calib_loader, quant_model)

    if args.gpfq:
        print("Performing GPFQ:")
        apply_gpfq(calib_loader, quant_model, p=args.gpfq_p, act_order=args.gpxq_act_order)

    if args.gpfa2q:
        print("Performing GPFA2Q:")
        apply_gpfq(
            calib_loader,
            quant_model,
            p=args.gpfq_p,
            act_order=args.gpxq_act_order,
            use_gpfa2q=args.gpfa2q,
            accumulator_bit_width=args.accumulator_bit_width)

    if args.gptq:
        print("Performing GPTQ:")
        apply_gptq(calib_loader, quant_model, act_order=args.gpxq_act_order)

    if args.learned_round:
        print("Applying Learned Round:")
        apply_learned_round_learning(
            quant_model,
            calib_loader,
            iters=args.learned_round_iters,
            optimizer_lr=args.learned_round_lr)

    if args.calibrate_bn:
        print("Calibrate BN:")
        calibrate_bn(calib_loader, quant_model)

    if args.bias_corr:
        print("Applying bias correction:")
        apply_bias_correction(calib_loader, quant_model)

    # Validate the quant_model on the validation dataloader
    print("Starting validation:")
    validate(val_loader, quant_model, stable=dtype != torch.bfloat16)

    if args.export_onnx_qcdq or args.export_torch_qcdq:
        # Generate reference input tensor to drive the export process
        model_config = get_model_config(args.model_name)
        center_crop_shape = model_config['center_crop_shape']
        img_shape = center_crop_shape
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        ref_input = torch.ones(1, 3, img_shape, img_shape, device=device, dtype=dtype)

        export_name = os.path.join(args.export_dir, config)
        if args.export_onnx_qcdq:
            export_name = export_name + '.onnx'
            export_onnx_qcdq(model, ref_input, export_name, opset_version=args.onnx_opset_version)
        if args.export_torch_qcdq:
            export_name = export_name + '.pt'
            export_torch_qcdq(model, ref_input, export_name)


if __name__ == '__main__':
    main()
