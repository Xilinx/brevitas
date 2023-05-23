# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import random
import warnings

from tqdm import tqdm
import numpy as np
import torch
import torchvision

from brevitas.dynamo.compile import brevitas_dynamo
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate
from brevitas_examples.imagenet_classification.ptq.utils import add_bool_arg


# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")


model_names = sorted(
    name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
    callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument(
    '--calibration-dir',
    default='/scratch/datasets/imagenet_symlink/val',
    #required=True,
    help='Path to folder containing Imagenet calibration folder')
parser.add_argument(
    '--validation-dir', 
    default='/scratch/datasets/imagenet_symlink/val',
    #required=True, 
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
    default=64,
    type=int,
    help='Minibatch size for validation (default: 64)')
parser.add_argument(
    '--validation-subset',
    default=1024,
    type=int,
    help='Subset size for validation (default: 1024)')
parser.add_argument(
    '--calibration-samples', default=256, type=int, help='Calibration size (default: 256)')
parser.add_argument(
    '--model-name',
    default='resnet18',
    metavar='ARCH',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use with PyTorch (default: None)')
parser.add_argument(
    '--quantization-backend',
    default='generic',
    choices=['generic', 'layerwise', 'flexml'],
    help='Backend to target for quantization (default: generic)')
parser.add_argument(
    '--compiler-backend',
    default='onnxrt_cpu',
    choices=['onnxrt_cpu', 'onnxrt_gpu'],
    help='Backend to target for quantization (default: generic)')
parser.add_argument(
    '--act-bit-width', default=8, type=int, help='Activations bit width (default: 8)')
parser.add_argument(
    '--weight-bit-width', default=8, type=int, help='Weights bit width (default: 8)')
parser.add_argument(
    '--act-quant-type',
    default='symmetric',
    choices=['symmetric', 'asymmetric'],
    help='Activation quantization type (default: symmetric)')
add_bool_arg(
    parser,
    'scaling-per-output-channel',
    default=True,
    help='Weight scaling per output channel (default: enabled)')
add_bool_arg(
    parser,
    'ptq-on-calibration-data',
    default=True,
    help='Run PTQ on a separate calibration dataloader (default: enabled)')

def main():
    args = parser.parse_args()

    # Set randomness
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
        inception_preprocessing=inception_preprocessing,
        subset_size=args.validation_subset)

    # Get the model from torchvision
    model = get_torchvision_model(args.model_name)
    model.eval()
    
    # If available, use the selected GPU for PyTorch execution
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Validate the floating point model 
    # on the validation dataloader
    print("Starting validation of the float model:")
    validate(val_loader, model)        
    
    # Create the brevitas dynamo backend
    brevitas_dynamo_backend = brevitas_dynamo(
            ptq_iters=len(calib_loader), 
            quantization_backend=args.quantization_backend, 
            weight_bit_width=args.weight_bit_width,
            act_bit_width=args.act_bit_width,
            act_quant_type=args.act_quant_type,
            scaling_per_output_channel=args.scaling_per_output_channel,
            compiler_backend=args.compiler_backend)
        
    # Pass the brevitas backend to torch compile
    model = torch.compile(model, backend=brevitas_dynamo_backend)
    
    # Perform PTQ on a standalone calibration dataset. 
    # Internally it's currently calling activation calibration and bias correction
    if args.ptq_on_calibration_data:
        print("Starting PTQ on calibration data:")
        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            for _, (images, _) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)

    # Validate the model on the validation dataloader
    # This is running in ONNXRuntime through dynamo
    print("Starting validation of the quant model running in ONNXRuntime through dynamo:")
    validate(val_loader, model)


if __name__ == '__main__':
    main()

