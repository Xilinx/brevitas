# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
import logging
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import pytest_cases
import torch
from torch.utils.data import Dataset

from brevitas_examples.imagenet_classification.ptq.ptq_evaluate import quantize_ptq_imagenet
from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import create_args_parser
from tests.brevitas_examples.common import assert_metrics
from tests.brevitas_examples.common import get_default_args
from tests.brevitas_examples.common import process_args_and_metrics
from tests.brevitas_examples.common import UpdatableNamespace
from tests.conftest import SEED

# TODO (pml): Use stricter tolerance once a proper image dataset is used.
# The validation set has 6 images, so ATOL_ACC is set to tolerate a single missclasification
# compared to the expected value.
ATOL_ACC = 1. / 6 * 100.
RTOL_ACC = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# TODO (pml): Include a real dataset for testing the ImageNet entrypoint.
def mock_generate_dataset(
        dir: str,
        resize_shape: int = 256,
        center_crop_shape: int = 224,
        inception_preprocessing: bool = False) -> Dataset:
    assert center_crop_shape == 224, "The labels for the test dataset were generated for tensors of size (3, 224, 224)."
    # The first element in the tuples in IMAGE_SEEDS_AND_LABELS represents the seed that was used
    # to generate a random tensor of size (3, 224, 224) to which the 'resnet18' model assigns the
    # label in the second element of the tuple. However, this approach limits ourselves to testing
    # only in 'resnet18'.
    IMAGE_SEEDS_AND_LABELS = [(55, 611), (0, 107), (66, 611), (1, 107), (95, 611), (2, 107)]
    return [(torch.randn((3, 224, 224), generator=torch.Generator().manual_seed(seed)), label)
            for seed,
            label in IMAGE_SEEDS_AND_LABELS]


@pytest.fixture()
def parser() -> ArgumentParser:
    return create_args_parser()


@pytest.fixture
def main() -> Callable:

    def wrapper_main(
            args: UpdatableNamespace,
            extra_args: Optional[List[str]] = None) -> Tuple[torch.nn.Module, Dict[str, float]]:
        # Mock dataset
        with patch('brevitas_examples.imagenet_classification.utils.generate_dataset',
                   mock_generate_dataset):
            results, model = quantize_ptq_imagenet(args, extra_args=extra_args)
        return results, model

    return wrapper_main


@pytest.fixture()
def default_run_args(parser: ArgumentParser):
    args = get_default_args(parser)
    args.calibration_dir = ""
    args.validation_dir = ""
    args.bias_corr = False
    args.batch_size_calibration = 2
    args.batch_size_validation = 2
    args.calibration_samples = 6
    args.validation_samples = 6
    return args


class ImageNetCases:

    METRICS = ["quant_top1"]

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model_name": "resnet18", "quant_top1": 50.0},
            {
                "model_name": "resnet18", "bias_corr": True, "quant_top1": 83.3333},
            {
                "model_name": "resnet18", "gptq": True, "quant_top1": 66.6667},],
        ids=["res-defaults", "res-bias-corr", "res-gptq"])
    def case_small_models_args_and_metrics(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(default_run_args, run_dict, extra_keys=ImageNetCases.METRICS)


@pytest.mark.vision
@pytest_cases.parametrize_with_cases("args_and_metrics", cases=ImageNetCases)
def test_quality_metrics(caplog, args_and_metrics, main):
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_and_metrics
    results, _ = main(args, extra_args)
    assert_metrics(results, exp_metrics, atol=ATOL_ACC, rtol=RTOL_ACC)
