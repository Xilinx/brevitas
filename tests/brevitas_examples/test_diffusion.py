# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
import logging
import os
import random
import shutil
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytest
import pytest_cases
import torch

from brevitas_examples.stable_diffusion.main import quantize_sd
from brevitas_examples.stable_diffusion.stable_diffusion_args import create_args_parser
from tests.brevitas_examples.common import assert_metrics
from tests.brevitas_examples.common import get_default_args
from tests.brevitas_examples.common import process_args_and_metrics
from tests.brevitas_examples.common import UpdatableNamespace
from tests.conftest import SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ATOL_FID = 1e-6
RTOL_FID = 1e-5


def ptid2pathname(string):
    return f'./{string.replace("/", "-").replace(":", "-")}'


@pytest.fixture()
def parser() -> ArgumentParser:
    return create_args_parser()


@pytest.fixture()
def default_run_args(parser: ArgumentParser, request):
    args = get_default_args(parser)
    args.output_path = ptid2pathname(request.node.nodeid)
    args.guidance_scale = 3.
    args.prompt = 2
    args.calibration_prompt = 1
    args.resolution = 64
    args.resolution = 64
    args.inference_steps = 10
    args.deterministic = True
    args.dtype = "float32"
    args.device = "cpu"
    return args


class StableDiffusionCases:

    METRICS = ["torchmetrics_fid", "clean_fid"]

    @pytest_cases.parametrize(
        "run_dict",
        [
            {
                "model": "hf-internal-testing/tiny-stable-diffusion-pipe",
                "torchmetrics_fid": 0.073824875056743625,
                "clean_fid": 0.0},
            {
                "model": "hf-internal-testing/tiny-stable-diffusion-pipe",
                "bias_correction": True,
                "torchmetrics_fid": 0.0747433751821518,
                "clean_fid": 0.0},
            {
                "model": "hf-internal-testing/tiny-stable-diffusion-pipe",
                "activation_equalization": True,
                "torchmetrics_fid": 0.05002012476325035,
                "clean_fid": 0.0},],
        ids=["sd-defaults", "sd-bias-corr", "sd-act-eq"])
    def case_small_models_args_and_metrics(self, run_dict, default_run_args, request):
        yield process_args_and_metrics(
            default_run_args, run_dict, extra_keys=StableDiffusionCases.METRICS)


def decorator_get_image_processor_dict(
        fun: Callable, image_processor_dict_overwrites: Dict[str, Any]) -> Callable:

    def wrap_get_image_processor_dict(*args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_processor_dict, kwargs = fun(*args, **kwargs)
        # Overwrite entries appropiately
        for key, value in image_processor_dict_overwrites.items():
            image_processor_dict[key] = value
        return image_processor_dict, kwargs

    return wrap_get_image_processor_dict


@pytest.fixture
def main() -> Callable:

    def wrapper_main(
            args: UpdatableNamespace,
            extra_args: Optional[List[str]] = None) -> Tuple[torch.nn.Module, Dict[str, float]]:
        if args.model == "hf-internal-testing/tiny-stable-diffusion-pipe":
            # Fix the configuration so the feature processor returns a tensor with the dimensions
            # expected by the safety checker
            from transformers.image_processing_base import ImageProcessingMixin
            ImageProcessingMixin.get_image_processor_dict = decorator_get_image_processor_dict(
                fun=ImageProcessingMixin.get_image_processor_dict,
                image_processor_dict_overwrites={
                    "crop_size": 30,
                    "size": 30,})
        # Create directory for storing the results
        os.makedirs(args.output_path, exist_ok=True)
        results, model = quantize_sd(args, extra_args=extra_args)
        # Clean-up after running the test
        shutil.rmtree(args.output_path)
        # Return the results along with the model
        return results, model

    return wrapper_main


@pytest.mark.diffusion
@pytest_cases.parametrize_with_cases("args_and_metrics", cases=StableDiffusionCases)
def test_quality_metrics(caplog, args_and_metrics, main):
    caplog.set_level(logging.INFO)
    args, extra_args, exp_metrics = args_and_metrics
    results, _ = main(args, extra_args)
    assert_metrics(results, exp_metrics, atol=ATOL_FID, rtol=RTOL_FID)
