# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from brevitas_examples.common.parse_utils import parse_args as parse_args_utils


class UpdatableNamespace(Namespace):

    def update(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)


def process_args_and_metrics(
        default_run_args: UpdatableNamespace, run_dict: Dict[str, Any],
        metric_keys: List[str]) -> Tuple[UpdatableNamespace, Dict[str, float]]:
    args = default_run_args
    exp_metrics = {}
    for metric in metric_keys:
        exp_metrics[metric] = run_dict[metric]
        del run_dict[metric]
    args.update(**run_dict)
    return args, exp_metrics


def get_default_args(parser: ArgumentParser) -> UpdatableNamespace:
    return UpdatableNamespace(**vars(parse_args_utils(parser, [])[0]))


# Check that all args in args are used
def parse_args_and_defaults(args: UpdatableNamespace,
                            parser: ArgumentParser) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    a = vars(args)
    da = vars(parse_args_utils(parser, [])[0])
    return a, da


def allclose(x, y, rtol, atol) -> bool:
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


def assert_test_args_and_metrics(
        main: Callable,
        args_and_metrics: Tuple[UpdatableNamespace, Dict[str, float]],
        atol: float,
        rtol: float) -> None:
    args, exp_metrics = args_and_metrics
    results, _ = main(args)
    # Evalute quality metrics
    for metric, exp_value in exp_metrics.items():
        value = results[metric]
        assert allclose(exp_value, value, rtol=rtol, atol=atol), f"Expected {metric} {exp_value}, measured {value}"
