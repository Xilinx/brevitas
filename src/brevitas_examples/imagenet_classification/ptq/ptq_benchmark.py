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
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_quant_model
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

config.IGNORE_MISSING_KEYS = True

# Torchvision models with top1 accuracy
TORCHVISION_TOP1_MAP = {
    'resnet18': 69.758,
    'mobilenet_v2': 71.898,
    'vit_b_32': 75.912,}

OPTIONS = {
    'model_name': TORCHVISION_TOP1_MAP.keys(),
    'target_backend': ['generic', 'layerwise', 'flexml'],  # Target backend
    'scale_factor_type': ['float32', 'po2'],  # Scale factor type
    'weight_bit_width': [8, 6, 4],  # Weight Bit Width
    'act_bit_width': [8],  # Act bit width
    'bias_bit_width': ['int32', 'int16'],  # Bias Bit-Width for Po2 scale
    'scaling_per_output_channel': [False, True],  # Scaling Per Output Channel
    'act_quant_type': ['asymmetric', 'symmetric'],  # Act Quant Type
    'bias_corr': [True],  # Bias Correction
    'graph_eq_iterations': [0, 20],  # Graph Equalization
    'graph_eq_merge_bias': [False, True],  # Merge bias for Graph Equalization
    'gptq': [False, True],  # Enable/Disable GPTQ
    'gptq_act_order': [False, True],  # Use act_order euristics for GPTQ
    'act_quant_percentile': [99.9, 99.99, 99.999],  # Activation Quantization Percentile
}

OPTIONS_DEFAULT = {
    'target_backend': ['generic'],  # Target backend
    'scale_factor_type': ['float32'],  # Scale factor type
    'weight_bit_width': [8],  # Weight Bit Width
    'act_bit_width': [8],  # Act bit width
    'bias_bit_width': ['int32'],  # Bias Bit-Width for Po2 scale
    'scaling_per_output_channel': [True],  # Scaling Per Output Channel
    'act_quant_type': ['symmetric'],  # Act Quant Type
    'bias_corr': [True],  # Bias Correction
    'graph_eq_iterations': [20],  # Graph Equalization
    'graph_eq_merge_bias': [True],  # Merge bias for Graph Equalization
    'gptq': [False],  # Enable/Disable GPTQ
    'gptq_act_order': [False],  # Use act_order euristics for GPTQ
    'act_quant_percentile': [99.999],  # Activation Quantization Percentile
}

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
parser.add_argument(
    '--options-to-exclude', choices=OPTIONS.keys(), nargs="+", default=[], help='Calibration size')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    for option in args.options_to_exclude:
        OPTIONS[option] = OPTIONS_DEFAULT[option]

    options_names = [k.replace('_', ' ').capitalize() for k in OPTIONS.keys()]
    torchvision_df = pd.DataFrame(
        columns=options_names + [
            'Top 1% floating point accuracy',
            'Top 1% quant accuracy',
            'Floating point accuracy - quant accuracy',
            'Quant accuracy / floating point accuracy',
            'Calibration size',
            'Calibration batch size',
            'Torch version',
            'Brevitas version'])

    ptq_torchvision_models(torchvision_df, args)

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


def ptq_torchvision_models(df, args):

    combinations = list(product(*OPTIONS.values()))

    k = 0
    for combination in combinations:
        config_namespace = SimpleNamespace()
        for key, value in zip(OPTIONS.keys(), combination):
            setattr(config_namespace, key, value)

        # Flexml supports only per-tensor scale factors, power of two scale factors
        if config_namespace.target_backend == 'flexml' and (
                config_namespace.scaling_per_output_channel or
                config_namespace.scale_factor_type == 'float32'):
            continue
        # Merge bias can be enabled only when graph equalization is enabled
        if config_namespace.graph_eq_iterations == 0 and config_namespace.graph_eq_merge_bias:
            continue
        # For generic and layerwise backend, we only test for int32 bias bit width
        if (config_namespace.target_backend == 'generic' or config_namespace.target_backend
                == 'layerwise') and config_namespace.bias_bit_width == 'int16':
            continue

        # If GPTQ is disabled, we do not care about the act_order heuristic
        if not config_namespace.gptq and config_namespace.gptq_act_order:
            continue

        fp_accuracy = TORCHVISION_TOP1_MAP[config_namespace.model_name]
        # Get model-specific configurations about input shapes and normalization
        model_config = get_model_config(config_namespace.model_name)

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
        model = get_torchvision_model(config_namespace.model_name)

        # Preprocess the model for quantization
        if config_namespace.target_backend == 'flexml':
            # Flexml requires static shapes, thus representative input is passed in
            img_shape = model_config['center_crop_shape']
            model = preprocess_for_flexml_quantize(
                model,
                torch.ones(1, 3, img_shape, img_shape),
                equalize_iters=config_namespace.graph_eq_iterations,
                equalize_merge_bias=config_namespace.graph_eq_merge_bias)
        elif config_namespace.target_backend == 'generic' or config_namespace.target_backend == 'layerwise':
            model = preprocess_for_quantize(
                model,
                equalize_iters=config_namespace.graph_eq_iterations,
                equalize_merge_bias=config_namespace.graph_eq_merge_bias)
        else:
            raise RuntimeError(f"{config_namespace.target_backend} backend not supported.")

        # Define the quantized model
        quant_model = quantize_model(
            model,
            backend=config_namespace.target_backend,
            act_bit_width=config_namespace.act_bit_width,
            weight_bit_width=config_namespace.weight_bit_width,
            bias_bit_width=config_namespace.bias_bit_width,
            scaling_per_output_channel=config_namespace.scaling_per_output_channel,
            act_quant_percentile=config_namespace.act_quant_percentile,
            act_quant_type=config_namespace.act_quant_type,
            scale_factor_type=config_namespace.scale_factor_type)

        # If available, use the selected GPU
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            quant_model = quant_model.cuda(args.gpu)
            cudnn.benchmark = False

        # Calibrate the quant_model on the calibration dataloader
        print("Starting calibration")
        calibrate(calib_loader, quant_model)

        if config_namespace.gptq:
            print("Performing gptq")
            apply_gptq(calib_loader, quant_model, config_namespace.gptq_act_order)

        if config_namespace.bias_corr:
            print("Applying bias correction")
            apply_bias_correction(calib_loader, quant_model)

        # Validate the quant_model on the validation dataloader
        print("Starting validation")
        top1 = validate(val_loader, quant_model)

        # Generate metrics for benchmarking
        top1 = np.around(top1, decimals=3)
        acc_diff = np.around(top1 - fp_accuracy, decimals=3)
        acc_ratio = np.around(top1 / fp_accuracy, decimals=3)

        df.at[k, :] = [v for _, v in config_namespace.__dict__.items()] + [
            fp_accuracy,
            top1,
            acc_diff,
            acc_ratio,
            args.calibration_samples,
            args.batch_size_calibration,
            torch_version,
            brevitas_version]

        df.to_csv('RESULTS_TORCHVISION.csv', index=False, mode='w')

        grouped_df = df.groupby([
            'Model name',
            'Target backend',
            'Scale factor type',
            'Weight bit width',
            'Act bit width',
            'Bias bit width',
            'Scaling per output channel',
            'Act quant type'])
        idx = grouped_df['Top 1% quant accuracy'].transform(max) == df['Top 1% quant accuracy']
        best_config_df = df[idx]
        best_config_df = best_config_df.sort_values(by=['Model name', 'Top 1% quant accuracy'])
        best_config_df.to_csv('RESULTS_TORCHVISION_BEST_CONFIGS.csv', index=False, mode='w')

        k += 1


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
        calibrate(calib_loader, model, bias_corr)

        # Validate the model on the validation dataloader
        top1 = validate(val_loader, model)

        # Generate metrics for benchmarking
        top1 = np.around(top1, decimals=3)
        acc_diff = np.around(top1 - fp_accuracy, decimals=3)
        acc_ratio = np.around(top1 / fp_accuracy, decimals=3)

        df.at[k, :] = [
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
