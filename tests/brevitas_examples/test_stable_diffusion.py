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

from brevitas_examples.common.parse_utils import parse_args as parse_args_utils
from brevitas_examples.stable_diffusion.main import quantize_sd
from brevitas_examples.stable_diffusion.stable_diffusion_args import create_args_parser

ATOL_PPL = 1e-6
RTOL_PPL = 1e-5

METRICS = ["torchmetrics_fid", "clean_fid"]


def parse_args(args):
    parser = create_args_parser()
    return parse_args_utils(parser, args)


def ptid2pathname(string):
    return f'./{string.replace("/", "-").replace(":", "-")}'


def allclose(x, y, rtol=RTOL_PPL, atol=ATOL_PPL):
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


# Check that all args in args are used
def validate_args(args):
    vars(args)
    vars(parse_args([])[0])


def decorator_get_image_processor_dict(
        fun: Callable, image_processor_dict_overwrites: Dict[str, Any]) -> Callable:

    def wrap_get_image_processor_dict(*args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_processor_dict, kwargs = fun(*args, **kwargs)
        # Overwrite entries appropiately
        for key, value in image_processor_dict_overwrites.items():
            image_processor_dict[key] = value
        return image_processor_dict, kwargs

    return wrap_get_image_processor_dict


def validate_args_and_run_main(args, extra_args=None):
    validate_args(args)
    # TODO (pml): Maybe refactor
    if args.model == "hf-internal-testing/tiny-stable-diffusion-pipe":
        # Fix the configuration so the feature processor returns a tensors with the dimensions
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
    # Clean-up after running th etest
    os.rmdir(args.output_path)
    return results, model


class UpdatableNamespace(Namespace):

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)


@pytest_cases.fixture()
def default_run_args(request):
    args = UpdatableNamespace(**vars(parse_args([])[0]))
    #args.checkpoint = ptid2pathname(request.node.nodeid) + ".pth" # Example filename which won't clash
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


@pytest_cases.fixture(
    ids=["sd-defaults", "sd-bias-corr", "sd-act-eq"],
    params=[
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
            "clean_fid": 0.0},])
def metrics_args_and_metrics(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    exp_metrics = {}
    for metric in METRICS:
        exp_metrics[metric] = run_dict[metric]
        del run_dict[metric]
    args.update(**run_dict)
    yield args, exp_metrics


@pytest.mark.sd
def test_small_models_metrics(caplog, metrics_args_and_metrics):
    caplog.set_level(logging.INFO)
    args, exp_metrics = metrics_args_and_metrics
    results, _ = validate_args_and_run_main(args)
    # Evalute quality metrics
    for metric, exp_value in exp_metrics.items():
        value = results[metric]
        assert allclose(exp_value, value), f"Expected {metric} {exp_value}, measured {value}"
