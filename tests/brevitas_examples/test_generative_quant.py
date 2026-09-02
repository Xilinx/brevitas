# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Apply every quantizer in WEIGHT_QUANT_MAP / INPUT_QUANT_MAP and run a forward pass.

For each combination of keys in ``WEIGHT_QUANT_MAP`` and ``INPUT_QUANT_MAP`` (defined in
``brevitas_examples/common/generative/quantize.py``), a Linear layer is quantized through
the same flow as the generative entrypoints (``generate_quantizers`` ->
``generate_quant_maps`` -> ``layerwise_quantize``) and a forward pass is run.

The map keys map onto ``generate_quantizers`` arguments as follows:

  WEIGHT_QUANT_MAP[fmt][scale_precision][param_method][granularity][quant_type]
      -> weight_quant_format, weight_scale_precision, weight_param_method,
         weight_quant_granularity, weight_quant_type

  INPUT_QUANT_MAP[fmt][scale_type][scale_precision][param_method][granularity][quant_type]
      -> input_quant_format, input_scale_type, input_scale_precision,
         input_param_method, input_quant_granularity, input_quant_type
  (the 'no_scale' branch has the shorter path [fmt][scale_type][quant_type])
"""

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import pytest
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

from brevitas import config
from brevitas.graph.quantize import layerwise_quantize
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.generative.quantize import INPUT_QUANT_MAP
from brevitas_examples.common.generative.quantize import WEIGHT_QUANT_MAP

IN_FEATURES: int = 32
OUT_FEATURES: int = 32
GROUP_SIZE: int = 8
WEIGHT_BIT_WIDTH: int = 8
INPUT_BIT_WIDTH: int = 8

# (id, quantizer kwargs) pair used to parametrize a single quant map combination.
Combination = Tuple[str, Dict[str, Any]]

# Weight scaling impl types used by the generative entrypoints.
WEIGHT_SCALING_IMPL_TYPES: List[str] = ['parameter_from_stats', 'stats']

# Param methods that rely on local loss (e.g. MSE, HQO), which require JIT to be disabled.
LOCAL_LOSS_PARAM_METHODS = {'mse', 'hqo'}


def _skip_if_incompatible_with_jit(kwargs: Dict[str, Any]) -> None:
    """Skip combinations that require JIT to be disabled when JIT is enabled."""
    if not config.JIT_ENABLED:
        return
    if kwargs.get("input_scale_type") == "dynamic":
        pytest.skip("Dynamic activation quantization requires JIT to be disabled")
    if kwargs.get("weight_param_method") in LOCAL_LOSS_PARAM_METHODS or \
            kwargs.get("input_param_method") in LOCAL_LOSS_PARAM_METHODS:
        pytest.skip("Local loss functions (e.g. MSE, HQO) require JIT to be disabled")


def _flatten_paths(d: Any, depth: int, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, ...]]:
    """Yield key-paths of exactly `depth` levels whose leaf is a quantizer class."""
    if depth == 0:
        if not isinstance(d, dict):
            yield prefix
        return
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        yield from _flatten_paths(v, depth - 1, prefix + (k,))


def _weight_combinations() -> List[Combination]:
    """Return (id, weight_kwargs) for each WEIGHT_QUANT_MAP leaf."""
    combos: List[Combination] = []
    # WEIGHT_QUANT_MAP[fmt][scale_precision][param_method][granularity][quant_type]
    for path in _flatten_paths(WEIGHT_QUANT_MAP, depth=5):
        fmt, scale_precision, param_method, granularity, quant_type = path
        kwargs = {
            "weight_quant_format": fmt,
            "weight_scale_precision": scale_precision,
            "weight_param_method": param_method,
            "weight_quant_granularity": granularity,
            "weight_quant_type": quant_type,}
        combos.append(("-".join(path), kwargs))
    return combos


def _input_combinations() -> List[Combination]:
    """
    Return (id, input_kwargs) for each INPUT_QUANT_MAP leaf.

    Two path shapes exist: the regular one and the shorter 'no_scale' one
    ([fmt][scale_type][quant_type]).
    """
    combos: List[Combination] = []
    for fmt, by_scale_type in INPUT_QUANT_MAP.items():
        for scale_type, sub in by_scale_type.items():
            if scale_type == "no_scale":
                # sub == {quant_type: cls}
                for quant_type in sub:
                    path = (fmt, scale_type, quant_type)
                    kwargs = {
                        "input_quant_format": fmt,
                        "input_scale_type": scale_type,
                        "input_scale_precision": None,
                        "input_param_method": None,
                        "input_quant_granularity": None,
                        "input_quant_type": quant_type,}
                    combos.append(("-".join(path), kwargs))
            else:
                # sub == [scale_precision][param_method][granularity][quant_type]
                for tail in _flatten_paths(sub, depth=4):
                    scale_precision, param_method, granularity, quant_type = tail
                    path = (fmt, scale_type) + tail
                    kwargs = {
                        "input_quant_format": fmt,
                        "input_scale_type": scale_type,
                        "input_scale_precision": scale_precision,
                        "input_param_method": param_method,
                        "input_quant_granularity": granularity,
                        "input_quant_type": quant_type,}
                    combos.append(("-".join(path), kwargs))
    return combos


WEIGHT_COMBINATIONS: List[Combination] = _weight_combinations()
INPUT_COMBINATIONS: List[Combination] = _input_combinations()


@pytest.fixture(scope="module")
def model() -> Tuple[Module, Tensor]:
    """Return a Linear layer and a small input batch."""
    torch.manual_seed(0)
    layer = nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float32)
    layer.eval()
    inp = torch.randn(4, IN_FEATURES)
    return layer, inp


def _generate_quantizers_kwargs(
        weight_kwargs: Dict[str, Any],
        input_kwargs: Optional[Dict[str, Any]],
        weight_scaling_impl_type: str = 'parameter_from_stats') -> Dict[str, Any]:
    """
    Assemble the generate_quantizers kwargs.

    Start from valid non-None defaults for every knob and override only the keys
    under test.
    """
    kwargs: Dict[str, Any] = dict(
        weight_bit_width=WEIGHT_BIT_WIDTH,
        weight_param_method='stats',
        weight_scale_precision='float_scale',
        weight_quant_type='sym',
        weight_quant_granularity='per_channel',
        weight_group_size=GROUP_SIZE,
        weight_scaling_impl_type=weight_scaling_impl_type,
        quantize_weight_zero_point=False,
        weight_quant_format='int',
        input_bit_width=None,
        input_quant_format='int',
        input_scale_precision='float_scale',
        input_scale_type='static',
        input_param_method='stats',
        input_quant_type='asym',
        input_quant_granularity='per_tensor',
        input_group_size=GROUP_SIZE,
        quantize_input_zero_point=False,
    )
    kwargs.update(weight_kwargs)
    if input_kwargs is not None:
        kwargs["input_bit_width"] = INPUT_BIT_WIDTH
        kwargs.update({k: v for k, v in input_kwargs.items() if v is not None})
    return kwargs


def _quantize_and_forward(
        model: Tuple[Module, Tensor],
        weight_kwargs: Dict[str, Any],
        input_kwargs: Optional[Dict[str, Any]],
        weight_scaling_impl_type: str = 'parameter_from_stats') -> Tensor:
    base_model, inp = model
    model = deepcopy(base_model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # generate_quantizers -> generate_quant_maps -> layerwise_quantize
    quantizers_dict = generate_quantizers(
        **_generate_quantizers_kwargs(weight_kwargs, input_kwargs, weight_scaling_impl_type))
    layer_map = generate_quant_maps(
        **quantizers_dict, dtype=dtype, device=device, quantize_embedding=False)
    model = layerwise_quantize(model=model, compute_layer_map=layer_map)

    model.eval()
    with torch.no_grad():
        out = model(inp)
    assert torch.isfinite(out).all(), "Non-finite values in output"
    return out


@pytest.mark.parametrize("weight_scaling_impl_type", WEIGHT_SCALING_IMPL_TYPES)
@pytest.mark.parametrize(
    "weight_kwargs", [kw for _, kw in WEIGHT_COMBINATIONS], ids=[i for i, _ in WEIGHT_COMBINATIONS])
def test_weight_quant_map(
        model: Tuple[Module, Tensor], weight_kwargs: Dict[str, Any],
        weight_scaling_impl_type: str) -> None:
    """Each WEIGHT_QUANT_MAP quantizer applies and runs a forward pass (weight-only)."""
    _skip_if_incompatible_with_jit(weight_kwargs)
    _quantize_and_forward(
        model, weight_kwargs, input_kwargs=None, weight_scaling_impl_type=weight_scaling_impl_type)


@pytest.mark.parametrize("weight_scaling_impl_type", WEIGHT_SCALING_IMPL_TYPES)
@pytest.mark.parametrize(
    "input_kwargs", [kw for _, kw in INPUT_COMBINATIONS], ids=[i for i, _ in INPUT_COMBINATIONS])
def test_input_quant_map(
        model: Tuple[Module, Tensor], input_kwargs: Dict[str, Any],
        weight_scaling_impl_type: str) -> None:
    """
    Each INPUT_QUANT_MAP quantizer applies and runs a forward pass.

    Paired with a fixed int/per_channel weight quantizer to isolate the input quantizer.
    """
    _skip_if_incompatible_with_jit(input_kwargs)
    weight_kwargs: Dict[str, Any] = {
        "weight_quant_format": "int",
        "weight_scale_precision": "float_scale",
        "weight_param_method": "stats",
        "weight_quant_granularity": "per_channel",
        "weight_quant_type": "sym",}
    _quantize_and_forward(
        model, weight_kwargs, input_kwargs, weight_scaling_impl_type=weight_scaling_impl_type)
