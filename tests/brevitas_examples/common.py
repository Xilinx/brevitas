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

# Scalar attribute types worth comparing between two modules of the same class.
# Tensors are covered by state_dict parity; callables (lambdas such as
# proxy_forward / dynamic_scaling_broadcastable_fn) differ by identity and are
# noise, so both are excluded.
_FINGERPRINT_SCALARS = (bool, int, float, str, type(None))


def _normalize_attr(value: Any) -> Any:
    """Normalize a scalar attribute for comparison: enums by their value,
    everything else unchanged. 'absent' is represented by the caller as None."""
    # Enum members expose `.value`; use it so two equal enums compare equal
    # regardless of identity / repr.
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not isinstance(value, _FINGERPRINT_SCALARS):
        return enum_value
    return value


# Attributes excluded from the fingerprint:
#   training          - execution state, flips with .train()/.eval()
#   stats_output_shape - a derived scalar-stats shape that is represented
#                        interchangeably as () or (1,); the actual stored shapes
#                        are already compared via state_dict parity.
_FINGERPRINT_IGNORE = {"training", "stats_output_shape"}


def _module_scalar_attrs(module: Module) -> Dict[str, Any]:
    """Public scalar (and scalar-tuple) attributes set on a module instance.

    None-valued attributes are dropped so that 'absent' and 'explicitly None'
    compare equal (e.g. update_state_dict_impl, present as None on one side and
    unset on the other, carry the same meaning)."""
    out: Dict[str, Any] = {}
    for key, value in vars(module).items():
        if key.startswith("_") or value is None or key in _FINGERPRINT_IGNORE:
            continue
        if isinstance(value, Module):
            continue
        if isinstance(value, _FINGERPRINT_SCALARS):
            out[key] = _normalize_attr(value)
        elif isinstance(value, tuple) and all(isinstance(e, _FINGERPRINT_SCALARS) for e in value):
            out[key] = value
    return out


def module_fingerprint(model: Module) -> List[Tuple[str, str, Tuple]]:
    """Ordered, comparable description of a model: for every submodule its name,
    fully-qualified type, and sorted scalar attributes.

    Tighter than a (name, type) hierarchy: two models match 1-to-1 only if they
    have the same submodules, in the same order, of the same types *and* with the
    same scalar attribute values (e.g. keepdim, quantize_zero_point, narrow_range).
    Tensor state is compared separately via state_dict parity.
    """
    fingerprint = []
    for name, module in model.named_modules():
        type_ = type(module)
        attrs = tuple(sorted(_module_scalar_attrs(module).items()))
        fingerprint.append((name, f"{type_.__module__}.{type_.__qualname__}", attrs))
    return fingerprint


def assert_state_dict_parity(ref: Module, built: Module) -> None:
    """Assert two models carry identical persistent state (keys + tensor values)."""
    ref_sd, built_sd = ref.state_dict(), built.state_dict()
    assert set(ref_sd) == set(built_sd), (
        f"state_dict keys differ: only in ref {sorted(set(ref_sd) - set(built_sd))}, "
        f"only in built {sorted(set(built_sd) - set(ref_sd))}")
    for key in ref_sd:
        assert ref_sd[key].shape == built_sd[key].shape and torch.equal(
            ref_sd[key], built_sd[key]), f"state_dict['{key}'] differs"


class MockProcess:
    """Mock multiprocessing.Process that runs the target synchronously.

    Used in benchmark tests to avoid spawning real subprocesses, making
    everything run in a single thread for easy debugging.
    """

    def __init__(self, target=None, args=(), kwargs=None):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


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
