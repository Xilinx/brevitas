# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from dataclasses import dataclass
import logging
import shutil

import numpy as np
import pytest

from brevitas_examples.llm.main import main
from brevitas_examples.llm.main import parse_args


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


def allclose(x, y):
    return np.allclose(x, y, rtol=1e-02, atol=5e-01, equal_nan=False)


def allexact(x, y):
    return np.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=False)


class UpdatableNamespace(Namespace):

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)


def requires_fx(args):
    return args.act_equalization == "fx" or args.weight_equalization or args.ln_affine_merge


@dataclass
class ModelAndPpl:
    name: str
    float_ppl: float
    supports_fx: bool


@pytest.fixture(
    scope="session",
    params=[
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            float_ppl=None,
            supports_fx=True,
        ),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-OPTForCausalLM",
            float_ppl=None,
            supports_fx=True,
        ),
        ModelAndPpl(
            name="hf-internal-testing/tiny-random-MistralForCausalLM",
            float_ppl=None,
            supports_fx=False,
        ),])
def small_models_with_ppl(request):
    yield request.param


@pytest.fixture()
def default_run_args(request):
    args = UpdatableNamespace(**vars(parse_args([])))
    args.nsamples = 2
    args.seqlen = 2
    args.model = "hf-internal-testing/tiny-random-MistralForCausalLM"
    args.dataset = "c4"
    args.eval = True
    #args.checkpoint = ptid2pathname(request.node.nodeid) + ".pth" # Example filename which won't clash
    args.weight_bit_width = 8
    args.weight_quant_granularity = "per_channel"  # "per_tensor", "per_channel", "per_group".
    args.input_bit_width = 8
    args.act_calibration = True
    return args


@pytest.fixture(
    params=[
        {},
        {
            "bias_corr": True},
        {
            "act_equalization": "layerwise"},
        {
            "act_equalization": "fx"},
        {
            "weight_equalization": True},
        {
            "gptq": True},
        {
            "ln_affine_merge": True},])
def toggle_run_args(default_run_args, request):
    args = default_run_args
    args.update(**request.param)
    yield args


@pytest.mark.examples
@pytest.mark.weekly
def test_small_models_toggle_run_args(caplog, toggle_run_args, small_models_with_ppl):
    caplog.set_level(logging.INFO)
    args = toggle_run_args
    args.model = small_models_with_ppl.name
    exp_float_ppl = small_models_with_ppl.float_ppl
    use_fx = requires_fx(args)
    if use_fx and not small_models_with_ppl.supports_fx:
        pytest.xfail(f"{small_models_with_ppl.name} does not support FX")
    float_ppl, quant_ppl, model = main(args)


@pytest.fixture(
    params=[
        {
            "model": "hf-internal-testing/tiny-random-MistralForCausalLM",
            "act_equalization": "layerwise",
            "gptq": True,
            "float_ppl": 31274.05078125,
            "quant_ppl": 33139.23046875},
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "act_equalization": "fx",
            "bias_corr": True,
            "float_ppl": 33239.5,
            "quant_ppl": 33283.75390625},
        {
            "model": "hf-internal-testing/tiny-random-OPTForCausalLM",
            "weight_equalization": True,
            "ln_affine_merge": True,
            "replace_mha": True,
            "float_ppl": 50016.0,
            "quant_ppl": 50016.0},])
def acc_args_and_acc(default_run_args, request):
    args = default_run_args
    run_dict = request.param
    float_ppl = run_dict["float_ppl"]
    quant_ppl = run_dict["quant_ppl"]
    del run_dict["float_ppl"]
    del run_dict["quant_ppl"]
    args.update(**run_dict)
    yield args, float_ppl, quant_ppl


@pytest.mark.examples
@pytest.mark.weekly
def test_small_models_acc(caplog, acc_args_and_acc):
    caplog.set_level(logging.INFO)
    args, exp_float_ppl, exp_quant_ppl = acc_args_and_acc
    float_ppl, quant_ppl, model = main(args)
    float_ppl = float_ppl.detach().cpu().numpy()
    quant_ppl = quant_ppl.detach().cpu().numpy()
    assert allexact(exp_float_ppl, float_ppl), f"Expected float PPL {exp_float_ppl}, measured PPL {float_ppl}"
    assert allexact(exp_quant_ppl, quant_ppl), f"Expected quant PPL {exp_quant_ppl}, measured PPL {quant_ppl}"
