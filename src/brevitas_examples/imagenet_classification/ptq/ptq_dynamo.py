# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import random
import warnings

from tqdm import tqdm
import numpy as np
import torch
import torchvision

from brevitas.dynamo.compile import brevitas_dynamo, brevitas_dynamo_ptq_mode
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

# Ignore warnings about __torch_function__
warnings.filterwarnings("ignore")

model_names = sorted(
    name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
    callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))

parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument(
    '--calibration-dir',
    required=True,
    help='Path to folder containing Imagenet calibration folder')
parser.add_argument(
    '--validation-dir', 
    required=True, 
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
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use (default: None)')
parser.add_argument(
    '--calibration-samples', default=256, type=int, help='Calibration size (default: 256)')
parser.add_argument(
    '--model-name',
    default='resnet18',
    metavar='ARCH',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument(
    '--target-backend',
    default='generic',
    choices=['generic', 'layerwise', 'flexml'],
    help='Backend to target for quantization (default: generic)')
parser.add_argument('--explicit-ptq', action='store_true', help="Call PTQ explicitly rather than within the dynamo backend")


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

    # Validate the floating point model 
    # on the validation dataloader
    print("Starting validation of the float model:")
    validate(val_loader, model)        
    
    # API option 1: PTQ is done explicitly by the user, more control, more verbose
    if args.explicit_ptq:
        brevitas_dynamo_backend = brevitas_dynamo(quantization_backend='generic', compiler_backend='onnxrt')
    # API option 2: PTQ is done implicitly with the brevitas_dynamo backend. Less control, less verbose
    else:
        brevitas_dynamo_backend = brevitas_dynamo(
            ptq_iters=len(calib_loader), quantization_backend='generic', compiler_backend='onnxrt')
        
    # Pass the brevitas backend to torch compile
    model = torch.compile(model, backend=brevitas_dynamo_backend)
    
    # Performing PTQ, either implicitly or explicitly
    
    # API option 1: PTQ is explicit and user controlled, we call a series of specific methods
    # Here we are calling only activation calibration and bias correction, but we could also do GPTQ, BN correction, etc.
    if args.explicit_ptq:
        
        # As long as we are under the brevitas_dynamo_ptq_mode context manager we delay compilation to the compiler backend
        # which is ONNXRuntime in this case. The moment we leave the context manager, forward triggers compilation.
        with brevitas_dynamo_ptq_mode():
            print("Starting PTQ calibration:")
            calibrate(calib_loader, model)
            print("Starting PTQ bias correction:")
            apply_bias_correction(calib_loader, model)
    
    # API option 2: PTQ is implicit and handled by brevitas_dynamo, we only call the model on calib data
    # Internally it's currently calling activation calibration and bias correction
    else:
        print("Starting PTQ:")
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
    print("Starting validation of the quant model:")
    validate(val_loader, model)


if __name__ == '__main__':
    main()

