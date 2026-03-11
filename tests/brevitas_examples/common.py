# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.nn import Module

from brevitas_examples.common.parse_utils import parse_args as parse_args_utils


class UpdatableNamespace(Namespace):

    def update(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)


def process_args_and_metrics(
    default_run_args: UpdatableNamespace,
    run_dict: Dict[str, Any],
    extra_keys: List[str] = None
) -> Tuple[UpdatableNamespace, Optional[List[str]], Dict[str, float]]:
    """
        Updates a copy of ``default_run_args`` with the values of ``run_dict``,
        after potentially removing the entry ``extra_args``, which corresponds
        to the keys that are not accepted by the entrypoint's parser, and those
        specified in ``extra_keys``, which represent the information needed
        to run the test, e.g. expected quality metrics for a given
        configuration.
    """
    # Dictionaries are copied to prevent interference with tests running
    # in parallel
    args = copy.copy(default_run_args)
    run_dict = copy.copy(run_dict)
    extra_args = None
    if "extra_args" in run_dict:
        extra_args = run_dict["extra_args"]
        del run_dict["extra_args"]
    exp_dict = {}
    if extra_keys is not None:
        for key in extra_keys:
            if key in run_dict:
                exp_dict[key] = run_dict[key]
                del run_dict[key]
    args.update(**run_dict)
    return args, extra_args, exp_dict


def get_default_args(parser: ArgumentParser) -> UpdatableNamespace:
    return UpdatableNamespace(**vars(parse_args_utils(parser, [])[0]))


def parse_args_and_defaults(args: UpdatableNamespace,
                            parser: ArgumentParser) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    a = vars(args)
    da = vars(parse_args_utils(parser, [])[0])
    return a, da


def allclose(x, y, rtol, atol) -> bool:
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


def assert_metrics(
        results: Dict[str, float],
        exp_metrics: Dict[str, float],
        atol: float,
        rtol: float,
        strict: bool = True) -> None:
    # Evalute quality metrics
    for metric, value in results.items():
        # If `strict=True`, all metrics in `results` are checked, so an error is raised
        # whenever there is a `metric` in `results` that is not registered `exp_metrics`
        if strict or metric in exp_metrics:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            exp_value = exp_metrics[metric]
            assert allclose(exp_value, value, rtol=rtol, atol=atol), f"Expected {metric} {exp_value}, measured {value}"


def assert_layer_types(model: Module, exp_layer_types: Dict[str, str]) -> None:
    for key, string in exp_layer_types.items():
        matched = False
        layer_names = []
        for name, layer in model.named_modules():
            layer_names += [name]
            if name == key:
                matched = True
                ltype = str(type(layer))
                assert ltype == string, f"Expected layer type: {string}, found {ltype} for key: {key}"
                continue
        assert matched, f"Layer key: {key} not found in {layer_names}"


def assert_layer_types_count(model: Module, exp_layer_types_count: Dict[str, int]) -> None:
    layer_types_count = {}
    for name, layer in model.named_modules():
        ltype = str(type(layer))
        if ltype not in layer_types_count:
            layer_types_count[ltype] = 0
        layer_types_count[ltype] += 1

    for name, count in exp_layer_types_count.items():
        curr_count = 0 if name not in layer_types_count else layer_types_count[name]
        assert count == curr_count, f"Expected {count} instances of layer type: {name}, found {curr_count}."
