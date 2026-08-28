# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply every quantizer in WEIGHT_QUANT_MAP / INPUT_QUANT_MAP and run a forward pass.

For each combination of keys in ``WEIGHT_QUANT_MAP`` and ``INPUT_QUANT_MAP``, a model is
quantized through the same flow as the LLM entrypoint (``generate_quantizers`` ->
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

from argparse import Namespace
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.nn import Module

from brevitas.graph.quantize import layerwise_quantize
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.generative.quantize import INPUT_QUANT_MAP
from brevitas_examples.common.generative.quantize import WEIGHT_QUANT_MAP
from brevitas_examples.llm.llm_args import create_args_parser
from tests.marker import jit_disabled_for_dynamic_quant_act
from tests.marker import requires_pt_ge

MODEL_IDS: List[str] = ["hf-internal-testing/tiny-random-LlamaForCausalLM"]

GROUP_SIZE: int = 8
WEIGHT_BIT_WIDTH: int = 8
INPUT_BIT_WIDTH: int = 8

# (id, quantizer kwargs) pair used to parametrize a single quant map combination.
Combination = Tuple[str, Dict[str, Any]]


def _entrypoint_arg_defaults() -> Namespace:
    """Return the LLM entrypoint's default args, to keep in sync with the entrypoint."""
    return create_args_parser().parse_args([])


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
    """Return (id, input_kwargs) for each INPUT_QUANT_MAP leaf.

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

# Weight scaling impl types used by the LLM entrypoint (see brevitas_examples/llm/main.py).
# Exercise both for the weight quantizer.
WEIGHT_SCALING_IMPL_TYPES: List[str] = ['parameter_from_stats', 'stats']


@pytest.fixture(scope="module", params=MODEL_IDS)
def model(request: pytest.FixtureRequest) -> Tuple[Module, Tensor]:
    """Load a model and a small input batch once per model id."""
    transformers = pytest.importorskip("transformers")

    model_id = request.param
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        input_ids = tokenizer("Brevitas quantization test.", return_tensors="pt")["input_ids"]
    except Exception:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    return model, input_ids


def _generate_quantizers_kwargs(
        weight_kwargs: Dict[str, Any],
        input_kwargs: Optional[Dict[str, Any]],
        weight_scaling_impl_type: str = 'parameter_from_stats') -> Dict[str, Any]:
    """Assemble the generate_quantizers kwargs from the entrypoint defaults.

    Start from the entrypoint's default args (so every input_* knob is a valid
    non-None value) and override only the keys under test.
    """
    args = _entrypoint_arg_defaults()
    # Same mapping args -> generate_quantizers as brevitas_examples/llm/main.py.
    kwargs = dict(
        weight_bit_width=WEIGHT_BIT_WIDTH,
        weight_param_method=args.weight_param_method,
        weight_scale_precision=args.weight_scale_precision,
        weight_quant_type=args.weight_quant_type,
        weight_quant_granularity=args.weight_quant_granularity,
        weight_group_size=GROUP_SIZE,
        weight_group_dim=args.weight_group_dim,
        weight_scaling_impl_type=weight_scaling_impl_type,
        quantize_weight_zero_point=args.quantize_weight_zero_point,
        weight_quant_format=args.weight_quant_format,
        input_bit_width=None,
        input_quant_format=args.input_quant_format,
        input_scale_precision=args.input_scale_precision,
        input_scale_type=args.input_scale_type,
        input_param_method=args.input_param_method,
        input_quant_type=args.input_quant_type,
        input_quant_granularity=args.input_quant_granularity,
        input_group_size=GROUP_SIZE,
        quantize_input_zero_point=args.quantize_input_zero_point,
        scale_rounding_func_type=args.scale_rounding_func_type,
        quant_attn_mode='sdpa',
        scaling_min_val=args.scaling_min_val,
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
    base_model, input_ids = model
    model = deepcopy(base_model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # Same flow as the LLM entrypoint (brevitas_examples/llm/main.py):
    #   generate_quantizers -> generate_quant_maps -> layerwise_quantize
    quantizers_dict = generate_quantizers(
        **_generate_quantizers_kwargs(weight_kwargs, input_kwargs, weight_scaling_impl_type))
    layer_map = generate_quant_maps(
        **quantizers_dict, dtype=dtype, device=device, quantize_embedding=False)
    model = layerwise_quantize(
        model=model, compute_layer_map=layer_map, name_blacklist=["lm_head", "embed_out"])

    model.eval()
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out
    assert torch.isfinite(logits).all(), "Non-finite values in output logits"
    return logits


@pytest.mark.llm
@requires_pt_ge('2.4')
@jit_disabled_for_dynamic_quant_act()
@pytest.mark.parametrize("weight_scaling_impl_type", WEIGHT_SCALING_IMPL_TYPES)
@pytest.mark.parametrize(
    "weight_kwargs", [kw for _, kw in WEIGHT_COMBINATIONS], ids=[i for i, _ in WEIGHT_COMBINATIONS])
def test_weight_quant_map(
        model: Tuple[Module, Tensor], weight_kwargs: Dict[str, Any],
        weight_scaling_impl_type: str) -> None:
    """Each WEIGHT_QUANT_MAP quantizer applies and runs a forward pass (weight-only)."""
    _quantize_and_forward(
        model, weight_kwargs, input_kwargs=None, weight_scaling_impl_type=weight_scaling_impl_type)


@pytest.mark.llm
@requires_pt_ge('2.4')
@jit_disabled_for_dynamic_quant_act()
@pytest.mark.parametrize("weight_scaling_impl_type", WEIGHT_SCALING_IMPL_TYPES)
@pytest.mark.parametrize(
    "input_kwargs", [kw for _, kw in INPUT_COMBINATIONS], ids=[i for i, _ in INPUT_COMBINATIONS])
def test_input_quant_map(
        model: Tuple[Module, Tensor], input_kwargs: Dict[str, Any],
        weight_scaling_impl_type: str) -> None:
    """Each INPUT_QUANT_MAP quantizer applies and runs a forward pass.

    Paired with a fixed int/per_channel weight quantizer to isolate the input quantizer.
    """
    weight_kwargs: Dict[str, Any] = {
        "weight_quant_format": "int",
        "weight_scale_precision": "float_scale",
        "weight_param_method": "stats",
        "weight_quant_granularity": "per_channel",
        "weight_quant_type": "sym",}
    _quantize_and_forward(
        model, weight_kwargs, input_kwargs, weight_scaling_impl_type=weight_scaling_impl_type)
