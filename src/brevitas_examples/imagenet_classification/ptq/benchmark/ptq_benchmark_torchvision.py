# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from functools import partial
from itertools import product
import os
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
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_act_equalization
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_bias_correction
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gpfq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_gptq
from brevitas_examples.imagenet_classification.ptq.ptq_common import apply_learned_round_learning
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas_examples.imagenet_classification.ptq.utils import get_gpu_index
from brevitas_examples.imagenet_classification.ptq.utils import get_model_config
from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model
from brevitas_examples.imagenet_classification.utils import generate_dataloader
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

config.IGNORE_MISSING_KEYS = True


def parse_type(v, default_type):
    if v == 'None':
        return None
    else:
        return default_type(v)


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class hashabledict(dict):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


# Torchvision models with top1 accuracy
TORCHVISION_TOP1_MAP = {
    'resnet18': 69.758,
    'mobilenet_v2': 71.898,
    'vit_b_32': 75.912,}

OPTIONS_DEFAULT = {
    'model_name': list(TORCHVISION_TOP1_MAP.keys()),
    'quant_format': ['int'],  # Quantization type (INT vs Float)
    'target_backend': ['layerwise'],  # Target backend
    'scale_factor_type': ['float_scale'],  # Scale factor type
    'weight_mantissa_bit_width': [4],
    'weight_exponent_bit_width': [3],
    'weight_narrow_range': [False],
    'layerwise_first_last_bit_width': [8],  # Input and weights bit width for first and last layer
    'act_mantissa_bit_width': [4],
    'act_exponent_bit_width': [3],
    'weight_bit_width': [8],  # Weight Bit Width
    'act_bit_width': [8],  # Act bit width
    'bias_bit_width': [32],  # Bias Bit-Width for Po2 scale
    'weight_quant_granularity': ['per_channel'],  # Scaling Per Output Channel
    'act_quant_type': ['sym'],  # Act Quant Type
    'act_param_method': ['stats'],  # Act Param Method
    'weight_param_method': ['mse'],  # Weight Quant Type
    'bias_corr': [True],  # Bias Correction
    'graph_eq_iterations': [20],  # Graph Equalization
    'graph_eq_merge_bias': [True],  # Merge bias for Graph Equalization
    'act_equalization': ['layerwise'],  # Perform Activation Equalization (Smoothquant)
    'learned_round': [False],  # Enable/Disable Learned Round
    'gptq': [False],  # Enable/Disable GPTQ
    'gpfq': [False],  # Enable/Disable GPFQ
    'gpfa2q': [False],  # Enable/Disable GPFA2Q
    'gpfq_p': [1.0],  # GPFQ P
    'gpxq_act_order': [False],  # Use act_order euristics for GPxQ
    'accumulator_bit_width': [16],  # Accumulator bit width, only in combination with GPFA2Q
    'act_quant_percentile': [99.999],  # Activation Quantization Percentile
    'uint_sym_act_for_unsigned_values': [True],  # Whether to use unsigned act quant when possible
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet PTQ Validation')
parser.add_argument('idx', type=int)
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
parser.add_argument('--calibration-samples', default=1000, type=int, help='Calibration size')
for option_name, option_value in OPTIONS_DEFAULT.items():
    if isinstance(option_value[0], bool):
        type_args = parse_bool
    else:
        type_args = partial(parse_type, default_type=type(option_value[0]))
    parser.add_argument(f'--{option_name}', default=option_value, nargs="+", type=type_args)


def main():
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args.gpu = get_gpu_index(args.idx)
    print("Iter {}, GPU {}".format(args.idx, args.gpu))

    try:
        ptq_torchvision_models(args)
    except Exception as E:
        print("Exception at index {}: {}".format(args.idx, E))


def ptq_torchvision_models(args):
    # Generate all possible combinations, including invalid ones

    options = {k: getattr(args, k) for k, _ in OPTIONS_DEFAULT.items()}

    combinations = list(product(*options.values()))

    configs = []
    for combination in combinations:
        config_namespace = SimpleNamespace(
            **{k: v for k, v in zip(OPTIONS_DEFAULT.keys(), combination)})
        config_namespace = validate_config(config_namespace)
        if config_namespace.is_valid:
            configs.append(hashabledict(**config_namespace.__dict__))

    configs = unique(configs)

    if args.idx > len(configs) - 1:
        return

    config_namespace = SimpleNamespace(**configs[args.idx])
    print(config_namespace)

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
    elif config_namespace.target_backend == 'fx' or config_namespace.target_backend == 'layerwise':
        model = preprocess_for_quantize(
            model,
            equalize_iters=config_namespace.graph_eq_iterations,
            equalize_merge_bias=config_namespace.graph_eq_merge_bias)
    else:
        raise RuntimeError(f"{config_namespace.target_backend} backend not supported.")

    if config_namespace.act_equalization is not None:
        print("Applying activation equalization:")
        apply_act_equalization(
            model, calib_loader, layerwise=config_namespace.act_equalization == 'layerwise')

    # Define the quantized model
    quant_model = quantize_model(
        model,
        quant_format=config_namespace.quant_format,
        backend=config_namespace.target_backend,
        act_bit_width=config_namespace.act_bit_width,
        layerwise_first_last_bit_width=config_namespace.layerwise_first_last_bit_width,
        weight_narrow_range=config_namespace.weight_narrow_range,
        weight_mantissa_bit_width=config_namespace.weight_mantissa_bit_width,
        weight_exponent_bit_width=config_namespace.weight_exponent_bit_width,
        act_mantissa_bit_width=config_namespace.act_mantissa_bit_width,
        act_exponent_bit_width=config_namespace.act_exponent_bit_width,
        weight_bit_width=config_namespace.weight_bit_width,
        weight_param_method=config_namespace.weight_param_method,
        act_param_method=config_namespace.act_param_method,
        bias_bit_width=config_namespace.bias_bit_width,
        weight_quant_granularity=config_namespace.weight_quant_granularity,
        act_quant_percentile=config_namespace.act_quant_percentile,
        act_quant_type=config_namespace.act_quant_type,
        scale_factor_type=config_namespace.scale_factor_type,
        uint_sym_act_for_unsigned_values=config_namespace.uint_sym_act_for_unsigned_values)

    # If available, use the selected GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        quant_model = quant_model.cuda(args.gpu)
        cudnn.benchmark = False

    # Calibrate the quant_model on the calibration dataloader
    print("Starting calibration")
    calibrate(calib_loader, quant_model)

    if config_namespace.gpfq:
        print("Performing GPFQ:")
        apply_gpfq(
            calib_loader,
            quant_model,
            p=config_namespace.gpfq_p,
            act_order=config_namespace.gpxq_act_order)

    if config_namespace.gpfa2q:
        print("Performing GPFA2Q:")
        apply_gpfq(
            calib_loader,
            quant_model,
            p=config_namespace.gpfq_p,
            act_order=config_namespace.gpxq_act_order,
            gpfa2q=config_namespace.gpfa2q,
            accumulator_bit_width=config_namespace.accumulator_bit_width)

    if config_namespace.gptq:
        print("Performing gptq")
        apply_gptq(calib_loader, quant_model, config_namespace.gpxq_act_order)

    if config_namespace.learned_round:
        print("Applying Learned Round:")
        apply_learned_round_learning(quant_model, calib_loader)

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

    column_names = [k.replace('_', ' ').capitalize() for k in config_namespace.__dict__.keys()] + [
        'Top 1% floating point accuracy',
        'Top 1% quant accuracy',
        'Floating point accuracy - quant accuracy',
        'Quant accuracy / floating point accuracy',
        'Calibration size',
        'Calibration batch size',
        'Torch version',
        'Brevitas version']
    values = [v for _, v in config_namespace.__dict__.items()] + [
        fp_accuracy,
        top1,
        acc_diff,
        acc_ratio,
        args.calibration_samples,
        args.batch_size_calibration,
        torch_version,
        brevitas_version]
    torchvision_df = pd.DataFrame([values], columns=column_names)

    folder = './multirun/' + str(args.idx)
    os.makedirs(folder, exist_ok=True)
    torchvision_df.to_csv(os.path.join(folder, 'RESULTS_TORCHVISION.csv'), index=False)


def validate_config(config_namespace):
    is_valid = True
    # Flexml supports only per-tensor scale factors, power of two scale factors
    if config_namespace.target_backend == 'flexml' and (
            config_namespace.weight_quant_granularity == 'per_channel' or
            config_namespace.scale_factor_type == 'float_scale'):
        is_valid = False
    # Merge bias can be enabled only when graph equalization is enabled
    if config_namespace.graph_eq_iterations == 0 and config_namespace.graph_eq_merge_bias:
        is_valid = False
    # For fx and layerwise backend, we only test for bias with bit width equals to 32
    if (config_namespace.target_backend == 'fx' or config_namespace.target_backend
            == 'layerwise') and config_namespace.bias_bit_width == 16:
        is_valid = False
    # Only one of GPTQ, GPFQ, or GPA2Q can be enabled, or none
    multiple_gpxqs = float(config_namespace.gpfq) + float(config_namespace.gptq) + float(
        config_namespace.gpfa2q)
    if multiple_gpxqs > 1:
        is_valid = False

    if config_namespace.act_equalization == 'layerwise' and config_namespace.target_backend == 'fx':
        is_valid = False
    if config_namespace.act_bit_width < config_namespace.weight_bit_width:
        is_valid = False

    if config_namespace.act_param_method == 'mse':
        config_namespace.act_quant_percentile = None
    # gpfq_p is needed for GPFQ and GPFA2Q
    if not config_namespace.gpfq and not config_namespace.gpfa2q:
        config_namespace.gpfq_p = None
    # accumulator bit width is not needed when not GPFA2Q
    if not config_namespace.gpfa2q:
        config_namespace.accumulator_bit_width = None

    if config_namespace.quant_format == 'int':
        config_namespace.weight_mantissa_bit_width = None
        config_namespace.weight_exponent_bit_width = None
        config_namespace.act_mantissa_bit_width = None
        config_namespace.act_exponent_bit_width = None

    if config_namespace.quant_format == 'float':
        config_namespace.act_quant_type = 'sym'
        config_namespace.weight_quant_type = 'sym'

    if config_namespace.quant_format == 'float':
        if config_namespace.weight_exponent_bit_width + config_namespace.weight_mantissa_bit_width != config_namespace.weight_bit_width - 1:
            is_valid = False
        if config_namespace.act_exponent_bit_width + config_namespace.act_mantissa_bit_width != config_namespace.act_bit_width - 1:
            is_valid = False

    config_namespace.is_valid = is_valid
    return config_namespace


if __name__ == '__main__':
    main()
