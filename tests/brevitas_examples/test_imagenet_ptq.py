# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import logging
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import pytest_cases
import torch
from torch.utils.data import Dataset

from brevitas_examples.common.parse_utils import parse_args as parse_args_utils
from brevitas_examples.imagenet_classification.ptq.ptq_evaluate import quantize_ptq_imagenet
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import create_args_parser
from tests.conftest import SEED

ATOL_PPL = 1e-4
RTOL_PPL = 1e-4

METRICS = ["quant_top1"]


def parse_args(args):
    parser = create_args_parser()
    return parse_args_utils(parser, args)


def allclose(x, y, rtol=RTOL_PPL, atol=ATOL_PPL):
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


# Check that all args in args are used
def validate_args(args):
    a = vars(args)
    da = vars(parse_args([])[0])


def mock_generate_dataset(
        dir: str,
        resize_shape: int = 256,
        center_crop_shape: int = 224,
        inception_preprocessing: bool = False) -> Dataset:
    assert center_crop_shape == 224, "The labels for the test dataset were generated for tensors of size (3, 224, 224)."
    # TODO (pml): Add explanation for this code
    IMAGE_SEEDS_AND_LABELS = [(55, 611), (0, 107), (66, 611), (1, 107), (95, 611), (2, 107)]
    return [(torch.randn((3, 224, 224), generator=torch.Generator().manual_seed(seed)), label)
            for seed,
            label in IMAGE_SEEDS_AND_LABELS]


def validate_args_and_run_main(args, extra_args=None):
    validate_args(args)
    with patch('brevitas_examples.imagenet_classification.utils.generate_dataset',
               mock_generate_dataset):
        results, model = quantize_ptq_imagenet(args, extra_args=extra_args)
    return results, model


class UpdatableNamespace(Namespace):

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)


@pytest_cases.fixture()
def default_run_args(request):
    args = UpdatableNamespace(**vars(parse_args([])[0]))
    #args.checkpoint = ptid2pathname(request.node.nodeid) + ".pth" # Example filename which won't clash
    args.calibration_dir = ""
    args.validation_dir = ""
    args.bias_corr = False
    args.batch_size_calibration = 2
    args.batch_size_validation = 2
    args.calibration_samples = 6
    args.validation_samples = 6
    return args


@pytest_cases.fixture(
    ids=["res-defaults", "res-bias-corr", "res-gptq"],
    params=[
        {
            "model_name": "resnet18", "quant_top1": 50.0},
        {
            "model_name": "resnet18", "bias_corr": True, "quant_top1": 66.6667},
        {
            "model_name": "resnet18", "gptq": True, "quant_top1": 83.3333},])
def metrics_args_and_metrics(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    exp_metrics = {}
    for metric in METRICS:
        exp_metrics[metric] = run_dict[metric]
        del run_dict[metric]
    args.update(**run_dict)
    yield args, exp_metrics


@pytest.mark.imagenet
def test_small_models_metrics(caplog, metrics_args_and_metrics):
    caplog.set_level(logging.INFO)
    args, exp_metrics = metrics_args_and_metrics
    results, _ = validate_args_and_run_main(args)
    # Evalute quality metrics
    for metric, exp_value in exp_metrics.items():
        value = results[metric]
        assert allclose(exp_value, value), f"Expected {metric} {exp_value}, measured {value}"
