# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
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


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(cfg):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    kwargs = OmegaConf.to_container(cfg)
    combination = SimpleNamespace()
    args = SimpleNamespace()
    for k, v in kwargs.items():
        if isinstance(v, dict):
            v = v['value']
        if k.isupper():
            k = k.lower()
            setattr(args, k, v)
        else:
            k = k.lower()
            setattr(combination, k, v)

    ptq_torchvision_models(args, combination)


def ptq_torchvision_models(args, combination):

    # Flexml supports only per-tensor scale factors, power of two scale factors
    if combination.target_backend == 'flexml' and (combination.scaling_per_output_channel or
                                                   combination.scale_factor_type == 'float32'):
        return
    # Merge bias can be enabled only when graph equalization is enabled
    if combination.graph_eq_iterations == 0 and combination.graph_eq_merge_bias:
        return
    # For generic and layerwise backend, we only test for int32 bias bit width
    if (combination.target_backend == 'generic' or
            combination.target_backend == 'layerwise') and combination.bias_bit_width == 'int16':
        return
    # If GPTQ is disabled, we do not care about the act_order heuristic
    if not combination.gptq and combination.gptq_act_order:
        return

    fp_accuracy = TORCHVISION_TOP1_MAP[combination.model_name]
    # Get model-specific configurations about input shapes and normalization
    model_config = get_model_config(combination.model_name)

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
    model = get_torchvision_model(combination.model_name)

    # Preprocess the model for quantization
    if combination.target_backend == 'flexml':
        # Flexml requires static shapes, thus representative input is passed in
        img_shape = model_config['center_crop_shape']
        model = preprocess_for_flexml_quantize(
            model,
            torch.ones(1, 3, img_shape, img_shape),
            equalize_iters=combination.graph_eq_iterations,
            equalize_merge_bias=combination.graph_eq_merge_bias)
    elif combination.target_backend == 'generic' or combination.target_backend == 'layerwise':
        model = preprocess_for_quantize(
            model,
            equalize_iters=combination.graph_eq_iterations,
            equalize_merge_bias=combination.graph_eq_merge_bias)
    else:
        raise RuntimeError(f"{combination.target_backend} backend not supported.")

    # Define the quantized model
    quant_model = quantize_model(
        model,
        backend=combination.target_backend,
        act_bit_width=combination.act_bit_width,
        weight_bit_width=combination.weight_bit_width,
        bias_bit_width=combination.bias_bit_width,
        scaling_per_output_channel=combination.scaling_per_output_channel,
        act_quant_percentile=combination.act_quant_percentile,
        act_quant_type=combination.act_quant_type,
        scale_factor_type=combination.scale_factor_type)

    # If available, use the selected GPU
    if args.gpu:
        quant_model = quant_model.cuda(int(os.environ["GPU_ID"]))

    # Calibrate the quant_model on the calibration dataloader
    print("Starting calibration")
    calibrate(calib_loader, quant_model)

    if combination.gptq:
        print("Performing gptq")
        apply_gptq(calib_loader, quant_model, combination.gptq_act_order)

    if combination.bias_corr:
        print("Applying bias correction")
        apply_bias_correction(calib_loader, quant_model)

    # Validate the quant_model on the validation dataloader
    print("Starting validation")
    top1 = validate(val_loader, quant_model)

    # Generate metrics for benchmarking
    top1 = np.around(top1, decimals=3)
    acc_diff = np.around(top1 - fp_accuracy, decimals=3)
    acc_ratio = np.around(top1 / fp_accuracy, decimals=3)

    options_names = [k.replace('_', ' ').capitalize() for k in combination.__dict__.keys()]
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
    torchvision_df.at[len(torchvision_df) - 1, :] = [v for _, v in combination.__dict__.items()] + [
        fp_accuracy,
        top1,
        acc_diff,
        acc_ratio,
        args.calibration_samples,
        args.batch_size_calibration,
        torch_version,
        brevitas_version]

    torchvision_df.to_csv('RESULTS_TORCHVISION.csv', index=False, mode='w')


if __name__ == '__main__':
    main()
