# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from itertools import product
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from brevitas import __version__ as brevitas_version
from brevitas import config
from brevitas import torch_version
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_quant_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

config.IGNORE_MISSING_KEYS = True

# Manually defined quantized model with original floating point accuracy
IMGCLSMOB_TOP1_MAP = {'quant_mobilenet_v1': 73.390}

parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument(
    '--calibration-dir',
    default='/scratch/datasets/imagenet_symlink/calibration',
    help='path to folder containing Imagenet calibration folder')
parser.add_argument(
    '--validation-dir',
    default='/scratch/datasets/imagenet_symlink/val',
    help='path to folder containing Imagenet validation folder')

parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
parser.add_argument(
    '--batch-size-calibration', default=64, type=int, help='Minibatch size for calibration')
parser.add_argument(
    '--batch-size-validation', default=256, type=int, help='Minibatch size for validation')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--calibration-samples', default=1000, type=int, help='Calibration size')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    quant_model_df = pd.DataFrame(
        columns=[
            'Model',
            'Bias correction',
            'Top 1% floating point accuracy',
            'Top 1% quant accuracy',
            'Floating point accuracy - quant accuracy',
            'Quant accuracy / floating point accuracy',
            'Calibration size',
            'Calibration batch size',
            'Torch version',
            'Brevitas version'])

    ptq_quant_models(quant_model_df, args)


def ptq_quant_models(df, args):
    options = [
        IMGCLSMOB_TOP1_MAP.keys(),
        [8],  #Bit width
        [True],  # Bias Correction
    ]

    combinations = list(product(*options))
    k = 0
    for i, (model_name, bit_width, bias_corr) in enumerate(combinations):
        fp_accuracy = IMGCLSMOB_TOP1_MAP[model_name]

        # Get model-specific configurations about input shapes and normalization
        model_config = get_model_config(model_name)

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
        model = get_quant_model(model_name, bit_width=bit_width)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            cudnn.benchmark = False

        # Calibrate the model on the calibration dataloader
        calibrate(calib_loader, model)

        if bias_corr:
            apply_bias_correction(calib_loader, model)

        # Validate the model on the validation dataloader
        top1 = validate(val_loader, model)

        # Generate metrics for benchmarking
        top1 = np.around(top1, decimals=3)
        acc_diff = np.around(top1 - fp_accuracy, decimals=3)
        acc_ratio = np.around(top1 / fp_accuracy, decimals=3)

        df.loc[k] = [
            model_name,
            bias_corr,
            fp_accuracy,
            top1,
            acc_diff,
            acc_ratio,
            args.calibration_samples,
            args.batch_size_calibration,
            torch_version,
            brevitas_version]
        k += 1
    df.to_csv('RESULTS_IMGCLSMOB.csv', index=False, mode='w')


if __name__ == '__main__':
    main()
