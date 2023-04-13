# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
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
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.ptq.ptq_common import validate
from brevitas_examples.imagenet_classification.ptq.utils import add_bool_arg
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED

# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")

model_names = sorted(
    name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
    callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))

parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument(
    '--calibration-dir',
    required=True,
    help='path to folder containing Imagenet calibration folder')
parser.add_argument(
    '--validation-dir', required=True, help='path to folder containing Imagenet validation folder')
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
    '--target-backend',
    default='generic',
    choices=['generic', 'flexml'],
    help='Backend to target for quantization (default: generic)')
parser.add_argument(
    '--scale-factor-type',
    default='float32',
    choices=['float32', 'po2'],
    help='Type for scale factors (default: float32)')
parser.add_argument(
    '--bit-width', default=8, type=int, help='Weights and activations bit width (default: 8)')
parser.add_argument(
    '--bias-bit-width',
    default='int32',
    choices=['int32', 'int16'],
    help='Bias bit width (default: int32)')
parser.add_argument(
    '--act-quant-type',
    default='symmetric',
    choices=['symmetric', 'asymmetric'],
    help='Activation quantization type (default: symmetric)')
parser.add_argument(
    '--graph-eq-iterations',
    default=20,
    type=int,
    help='Numbers of iterations for graph equalization (default: 20)')
parser.add_argument(
    '--act-quant-percentile',
    default=99.999,
    type=float,
    help='Percentile to use for stats of activation quantization (default: 99.999)')
parser.add_argument(
    '--export-path-onnx-qcdq',
    default=None,
    type=str,
    help='If specified, path where to export the model in onnx qcdq format')
parser.add_argument(
    '--export-path-torch-qcdq',
    default=None,
    type=str,
    help='If specified, path where to export the model in torch qcdq format (default: none)')
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


def main():
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(
        f"Model {args.model_name} - "
        f"Target backend: {args.target_backend} - "
        f"Quantization type: {args.scale_factor_type} - "
        f"Activation/Weight bit width {args.bit_width} - "
        f"Bias bit width: {args.bias_bit_width} - "
        f"Per-channel scale factors: {args.scaling_per_output_channel} - "
        f"Activation quant type: {args.act_quant_type} - "
        f"Bias Correction Enabled: {args.bias_corr} - "
        f"Iterations for graph equalization: {args.graph_eq_iterations} - "
        f"Merge bias in graph equalization: {args.graph_eq_merge_bias} - "
        f"Activation Quant Momentum: {args.act_quant_percentile}")

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

    # Preprocess the model for quantization
    if args.target_backend == 'flexml':
        # flexml requires static shapes, pass a representative input in
        img_shape = model_config['center_crop_shape']
        model = preprocess_for_flexml_quantize(
            model,
            torch.ones(1, 3, img_shape, img_shape),
            equalize_iters=args.equalize_iters,
            equalize_merge_bias=args.equalize_merge_bias)
    elif args.target_backend == 'generic':
        model = preprocess_for_quantize(
            model,
            equalize_iters=args.graph_eq_iterations,
            equalize_merge_bias=args.graph_eq_merge_bias)
    else:
        raise RuntimeError(f"{args.target_backend} backend not supported.")

    # Define the quantized model
    quant_model = quantize_model(
        model,
        backend=args.target_backend,
        bit_width=args.bit_width,
        bias_bit_width=args.bias_bit_width,
        scaling_per_output_channel=args.scaling_per_output_channel,
        act_quant_percentile=args.act_quant_percentile,
        act_quant_type=args.act_quant_type,
        scale_factor_type=args.scale_factor_type)

    # If available, use the selected GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        quant_model = quant_model.cuda(args.gpu)
        cudnn.benchmark = False

    # Calibrate the quant_model on the calibration dataloader
    print("Starting calibration")
    calibrate(calib_loader, quant_model, args.bias_corr)

    # Validate the quant_model on the validation dataloader
    print("Starting validation")
    validate(val_loader, quant_model)

    # Generate reference input tensor to drive the export process
    model_config = get_model_config(args.model_name)
    center_crop_shape = model_config['center_crop_shape']
    img_shape = center_crop_shape
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    ref_input = torch.ones(1, 3, img_shape, img_shape, device=device, dtype=dtype)
    if args.export_path_onnx_qcdq is not None:
        export_onnx_qcdq(model, ref_input, args.export_path_onnx_qcdq)
    if args.export_path_torch_qcdq is not None:
        export_torch_qcdq(model, ref_input, args.export_path_torch_qcdq)


if __name__ == '__main__':
    main()
