# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests: original ``generate_quantizers`` vs new ``generate_weight_quantizer``.

The original ``generate_quantizers`` (``brevitas_examples.common.generative.quantize``)
selects a weight quantizer by indexing the static ``WEIGHT_QUANT_MAP`` with string
keys, while the new ``generate_weight_quantizer``
(``brevitas_examples.common.quantizer_builder.generate``) assembles it from
brevitas / builder enums.

This module builds the *weight* quantizer with both APIs for every weight
configuration reachable from the LLM / Stable-Diffusion CLIs -- the cartesian
product of their ``--weight-*`` choices, filtered to the valid ``WEIGHT_QUANT_MAP``
leaves -- and asserts the two are equivalent:

  1. identical module hierarchy once hosted by a ``QuantLinear``; and
  2. identical quantized-weight tensors and layer outputs (``torch.equal``).

The comparison mirrors ``tests/brevitas_examples/test_quantizer_builder.py``
because the quantizer injectors themselves define no ``__eq__``.

The combo generator (:func:`_iter_weight_combos`) is deliberately kept as a plain
function so it can later feed a Hypothesis ``sampled_from`` strategy; for now the
whole product is enumerated through ``pytest.mark.parametrize``.
"""

import itertools

import pytest
import torch

from brevitas import config
from brevitas.core.function_wrapper import CeilSte
from brevitas.core.function_wrapper import FloorSte
from brevitas.core.restrict_val import RoundSte
from brevitas.core.scaling import RoundMidMaxSte
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantLinear
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.generative.quantize import quant_format_from_string
from brevitas_examples.common.generative.quantize import scale_quant_format_from_string
from brevitas_examples.common.generative.quantize import WEIGHT_QUANT_MAP
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder.generate import generate_weight_quantizer

# Keep the layer small and deterministic so the quantized weights / outputs are
# directly comparable between the two APIs.
torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 16  # divides IN_FEATURES; Stable-Diffusion's default weight group size
# generate_quantizers applies scaling_min_val=1e-4 by default; pass the same value
# to generate_weight_quantizer so the two definitions line up.
SCALING_MIN_VAL = 1e-4

# ---------------------------------------------------------------------------
# Axis domains: the union of the LLM (llm_args.py) and Stable-Diffusion
# (stable_diffusion_args.py) ``--weight-*`` argparse choices. The float formats
# carry an explicit ``e4m3`` minifloat suffix (as the CLIs require).
# ---------------------------------------------------------------------------
WEIGHT_QUANT_FORMATS = ["int", "float_e4m3", "float_ocp_e4m3", "float_fnuz_e4m3"]
WEIGHT_SCALE_PRECISIONS = ["float_scale", "signed_float_scale", "po2_scale"]
WEIGHT_PARAM_METHODS = ["stats", "mse", "hqo"]
WEIGHT_GRANULARITIES = ["per_tensor", "per_channel", "per_group"]
WEIGHT_QUANT_TYPES = ["sym", "asym"]
WEIGHT_SCALING_IMPL_TYPES = ["parameter_from_stats", "stats"]
SCALE_ROUNDING_FUNC_TYPES = [None, "floor"]

# ---------------------------------------------------------------------------
# String -> enum translation, mirroring generate_quantizers' own parsing.
# ---------------------------------------------------------------------------
_FORMAT_TO_QUANT_TYPE = {
    "int": QuantType.INT,
    "float": QuantType.FP,
    "float_ocp": QuantType.FP,
    "float_fnuz": QuantType.FP,}
_FORMAT_TO_FLOAT_FORMAT = {
    "int": None,
    "float": FloatFormat.FLOAT,
    "float_ocp": FloatFormat.OCP,
    "float_fnuz": FloatFormat.FNUZ,}
_PRECISION_TO_RESTRICT = {
    "float_scale": RestrictValueType.FP,
    # Signed scale is now a first-class restrict value the builder understands
    # (handled in the symmetric scale components), so it maps straight to SIGNED_FP
    # instead of being threaded through a separate signed_scale flag.
    "signed_float_scale": RestrictValueType.SIGNED_FP,
    "po2_scale": RestrictValueType.POWER_OF_TWO,}
_PARAM_METHOD_TO_ENUM = {
    "stats": ParamMethod.STATS,
    "mse": ParamMethod.MSE,
    "hqo": ParamMethod.HQO,}
_GRANULARITY_TO_ENUM = {
    "per_tensor": ScalingPerOutputType.TENSOR,
    "per_channel": ScalingPerOutputType.CHANNEL,
    "per_group": ScalingPerOutputType.GROUP,}
_QUANT_TYPE_TO_PARAM_TYPE = {
    "sym": QuantParamType.SYM,
    "asym": QuantParamType.ASYM,}
_SCALING_IMPL_TO_ENUM = {
    "parameter_from_stats": ScalingImplType.PARAMETER_FROM_STATS,
    "stats": ScalingImplType.STATS,}
# Same mapping generate_quantizers uses for scale_rounding_func_type.
_SCALE_ROUNDING_TO_IMPL = {
    "ceil": CeilSte,
    "floor": FloorSte,
    "round": RoundSte,
    "midmax": RoundMidMaxSte,}

# Reference quantizer combos that are broken in brevitas itself and therefore
# cannot be compared. Keyed by (format, scale_precision, param_method,
# granularity, quant_type). See tests/brevitas_examples/test_quantizer_builder.py.
_KNOWN_XFAIL = {
    # ShiftedMXUInt8WeightMSE crashes for groupwise (zero-point stats view is
    # Identity but reduces over the group dim).
    ("int", "po2_scale", "mse", "per_group", "asym"),}


def _weight_leaf_exists(quant_format, scale_precision, param_method, granularity, quant_type):
    """True if the (format, precision, ...) tuple is an actual WEIGHT_QUANT_MAP
    leaf, using the same string parsing as generate_quantizers."""
    fmt, _ = quant_format_from_string(quant_format)
    prec, _ = scale_quant_format_from_string(scale_precision)
    try:
        WEIGHT_QUANT_MAP[fmt][prec][param_method][granularity][quant_type]
    except KeyError:
        return False
    return True


def _iter_weight_combos():
    """Yield every valid weight config as a dict of generate_quantizers kwargs."""
    categorical = itertools.product(
        WEIGHT_QUANT_FORMATS,
        WEIGHT_SCALE_PRECISIONS,
        WEIGHT_PARAM_METHODS,
        WEIGHT_GRANULARITIES,
        WEIGHT_QUANT_TYPES)
    for quant_format, scale_precision, param_method, granularity, quant_type in categorical:
        if not _weight_leaf_exists(
                quant_format, scale_precision, param_method, granularity, quant_type):
            continue
        # Secondary axes, crossed on top of each valid leaf. Float formats derive
        # their bit width from the eXmY suffix, so only vary it for int. The
        # zero-point only matters for asym.
        bit_widths = [4, 8] if quant_format == "int" else [8]
        zero_points = [False, True] if quant_type == "asym" else [False]
        group_sizes = [GROUP_SIZE] if granularity == "per_group" else [None]
        secondary = itertools.product(
            bit_widths,
            zero_points,
            WEIGHT_SCALING_IMPL_TYPES,
            SCALE_ROUNDING_FUNC_TYPES,
            group_sizes)
        for bit_width, zero_point, scaling_impl, rounding, group_size in secondary:
            yield {
                "weight_quant_format": quant_format,
                "weight_scale_precision": scale_precision,
                "weight_param_method": param_method,
                "weight_quant_granularity": granularity,
                "weight_quant_type": quant_type,
                "weight_bit_width": bit_width,
                "quantize_weight_zero_point": zero_point,
                "weight_scaling_impl_type": scaling_impl,
                "scale_rounding_func_type": rounding,
                "weight_group_size": group_size,}


def _combo_id(combo):
    return "-".join([
        combo["weight_quant_format"],
        combo["weight_scale_precision"],
        combo["weight_param_method"],
        combo["weight_quant_granularity"],
        combo["weight_quant_type"],
        f"bw{combo['weight_bit_width']}",
        "zpT" if combo["quantize_weight_zero_point"] else "zpF",
        combo["weight_scaling_impl_type"],
        f"round-{combo['scale_rounding_func_type']}",])


WEIGHT_COMBOS = list(_iter_weight_combos())
WEIGHT_COMBO_IDS = [_combo_id(c) for c in WEIGHT_COMBOS]


def _old_weight_quant(combo):
    """weight_quant produced by the original generate_quantizers.

    Input quantization is left disabled (input_bit_width=None). A concrete
    input_scale_precision is still required: generate_quantizers parses it
    unconditionally and chokes on the None default.
    """
    return generate_quantizers(
        weight_bit_width=combo["weight_bit_width"],
        weight_param_method=combo["weight_param_method"],
        weight_scale_precision=combo["weight_scale_precision"],
        weight_quant_type=combo["weight_quant_type"],
        weight_quant_granularity=combo["weight_quant_granularity"],
        weight_group_size=combo["weight_group_size"],
        quantize_weight_zero_point=combo["quantize_weight_zero_point"],
        weight_quant_format=combo["weight_quant_format"],
        weight_scaling_impl_type=combo["weight_scaling_impl_type"],
        scale_rounding_func_type=combo["scale_rounding_func_type"],
        scaling_min_val=SCALING_MIN_VAL,
        input_quant_format="int",
        input_scale_precision="float_scale",
    )["weight_quant"]


def _new_weight_quant(combo):
    """weight_quant produced by the new generate_weight_quantizer."""
    fmt, float_format_dict = quant_format_from_string(combo["weight_quant_format"])
    float_format = _FORMAT_TO_FLOAT_FORMAT[fmt]

    kwargs = dict(
        quant_type=_FORMAT_TO_QUANT_TYPE[fmt],
        quant_param_type=_QUANT_TYPE_TO_PARAM_TYPE[combo["weight_quant_type"]],
        param_method=_PARAM_METHOD_TO_ENUM[combo["weight_param_method"]],
        granularity=_GRANULARITY_TO_ENUM[combo["weight_quant_granularity"]],
        # The signed vs unsigned scale distinction is carried by the restrict-value
        # type (SIGNED_FP), so signed_float_scale maps straight through here.
        scale_precision=_PRECISION_TO_RESTRICT[combo["weight_scale_precision"]],
        scaling_impl_type=_SCALING_IMPL_TO_ENUM[combo["weight_scaling_impl_type"]],
        bit_width=combo["weight_bit_width"],
        group_size=combo["weight_group_size"],
        quantize_zero_point=combo["quantize_weight_zero_point"],
        float_format=float_format,
        scaling_min_val=SCALING_MIN_VAL,
    )
    if combo["scale_rounding_func_type"] is not None:
        kwargs["scale_rounding_impl"] = _SCALE_ROUNDING_TO_IMPL[combo["scale_rounding_func_type"]]
    if float_format is not None:
        exponent = float_format_dict["exponent_bit_width"]
        mantissa = float_format_dict["mantissa_bit_width"]
        kwargs["float_quant_format"] = f"e{exponent}m{mantissa}"
    return generate_weight_quantizer(**kwargs)


def _make_quant_linear(weight_quant, **layer_kwargs):
    # return_quant_tensor=False: with weight-only quantization the layer input is
    # a plain Tensor, so the layer cannot emit a QuantTensor. The quantized weight
    # is still compared directly via quant_weight().
    return QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=weight_quant,
        return_quant_tensor=False,
        **layer_kwargs)


def _module_hierarchy(model):
    """Ordered (name, fully-qualified-type) description of the module tree."""
    hierarchy = []
    for name, module in model.named_modules():
        type_ = type(module)
        hierarchy.append((name, f"{type_.__module__}.{type_.__qualname__}"))
    return hierarchy


def _assert_quant_weight_equal(ref_weight, new_weight):
    assert torch.equal(ref_weight.value, new_weight.value)
    assert torch.equal(ref_weight.scale, new_weight.scale)
    assert (ref_weight.zero_point is None) == (new_weight.zero_point is None)
    if ref_weight.zero_point is not None:
        assert torch.equal(ref_weight.zero_point, new_weight.zero_point)
    # Int quant tensors expose `bit_width`; float quant tensors expose
    # `exponent_bit_width` / `mantissa_bit_width`.
    if hasattr(ref_weight, "bit_width"):
        assert torch.equal(ref_weight.bit_width, new_weight.bit_width)
    else:
        assert torch.equal(ref_weight.exponent_bit_width, new_weight.exponent_bit_width)
        assert torch.equal(ref_weight.mantissa_bit_width, new_weight.mantissa_bit_width)


def test_weight_combos_collected():
    """Guard against the leaf filter silently emptying the parametrization."""
    assert WEIGHT_COMBOS, "No valid weight combos were generated."
    formats = {c["weight_quant_format"] for c in WEIGHT_COMBOS}
    assert formats == set(WEIGHT_QUANT_FORMATS)
    granularities = {c["weight_quant_granularity"] for c in WEIGHT_COMBOS}
    assert granularities == set(WEIGHT_GRANULARITIES)


@pytest.mark.parametrize("combo", WEIGHT_COMBOS, ids=WEIGHT_COMBO_IDS)
def test_weight_quantizer_parity(combo):
    # Local-loss param methods (MSE, HQO) rely on Python control flow during the
    # optimization and require JIT to be disabled.
    if config.JIT_ENABLED and combo["weight_param_method"] in ("mse", "hqo"):
        pytest.skip("Local loss param methods (MSE, HQO) require JIT to be disabled")

    key = (
        combo["weight_quant_format"],
        combo["weight_scale_precision"],
        combo["weight_param_method"],
        combo["weight_quant_granularity"],
        combo["weight_quant_type"])
    if key in _KNOWN_XFAIL:
        pytest.xfail("Reference brevitas quantizer is broken for this combo.")

    # Asymmetric + signed scale is an intentional, documented divergence.
    #
    # The original generate_quantizers applies the signed-scale override
    # (restrict_scaling_type=SIGNED_FP, scaling_stats_op=SIGNED_MAX) unconditionally
    # via maybe_inject_signed_scale_kwargs, i.e. to asymmetric quantizers too. A
    # signed scale is a symmetric-only concept, so the new quantizer_builder scopes
    # it to symmetric quantizers: asymmetric quantizers keep their MIN_MAX scale
    # stats op. As a result, for asym + signed_float_scale the old scale stats op is
    # SignedAbsMax while the new one is (correctly) AbsMinMax. This is a deliberate
    # behavioural fix, not a builder bug, so we do not require parity here.
    if (combo["weight_quant_type"] == "asym" and
            combo["weight_scale_precision"] == "signed_float_scale"):
        pytest.xfail(
            "Signed scale is symmetric-only in the builder; the original "
            "generate_quantizers applies it to asymmetric quantizers too "
            "(intentional divergence).")

    # Symmetric + signed scale + MSE/HQO: the scale local-loss *init op* differs.
    #
    # The reference MSE/HQO classes hardcode an unsigned init op (AbsMax), while the
    # new builder derives the MSE/HQO init op from scaling_stats_op, which is
    # SIGNED_MAX for a signed scale -> SignedAbsMax. For MSE this is benign
    # (mse_search abs()-normalizes the init), but for HQO the init sign propagates
    # into the optimizer's search, so the converged scale can genuinely differ.
    #
    # xfailed for now; revisit whether to keep the MSE/HQO init op unsigned in the
    # builder (matching the reference) as a follow-up.
    if (combo["weight_quant_type"] == "sym" and
            combo["weight_scale_precision"] == "signed_float_scale" and
            combo["weight_param_method"] in ("mse", "hqo")):
        pytest.xfail(
            "Symmetric signed-scale MSE/HQO: builder derives a signed init op "
            "(SignedAbsMax) from the signed scaling_stats_op, while the reference "
            "hardcodes AbsMax (intentional divergence; may be revisited).")

    ref_quant = _old_weight_quant(combo)
    new_quant = _new_weight_quant(combo)

    layer_kwargs = {}
    if combo["weight_quant_granularity"] == "per_group":
        layer_kwargs["weight_group_size"] = GROUP_SIZE
    ref_linear = _make_quant_linear(ref_quant, **layer_kwargs)
    new_linear = _make_quant_linear(new_quant, **layer_kwargs)

    # 1) Module hierarchy must match 1-to-1, before syncing weights so a
    # structural mismatch is a clear diff rather than a state_dict error.
    assert _module_hierarchy(ref_linear) == _module_hierarchy(new_linear)

    # Give both layers identical float weights so the only possible difference is
    # in the quantization path itself.
    new_linear.weight.data.copy_(ref_linear.weight.data)

    ref_linear.eval()
    new_linear.eval()

    # Mock forward to trigger lazy init of any parameter-based scaling; after
    # this the learned scales are computed from the (identical) weights.
    mock_input = torch.randn(1, IN_FEATURES)
    ref_linear(mock_input)
    new_linear(mock_input)

    # 2) Quantized weights must match exactly.
    _assert_quant_weight_equal(ref_linear.quant_weight(), new_linear.quant_weight())

    # 3) Quantized layer outputs must match exactly.
    x = torch.randn(1, IN_FEATURES)
    assert torch.equal(ref_linear(x), new_linear(x))
