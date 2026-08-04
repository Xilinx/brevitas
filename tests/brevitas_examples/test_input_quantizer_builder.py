# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the input/activation quantizer builder.

Covers the ``static`` and ``dynamic`` scale types of ``INPUT_QUANT_MAP`` across
per_tensor / per_row / per_group granularities (the ``no_scale`` scale type is
not implemented in the builder yet). All specs are hosted by a ``QuantIdentity``;
per_row / per_group additionally inject the attributes a weight layer would
normally supply (per-channel broadcastable shape / group_dim) via ``.let()``.
"""

import pytest
import torch

from brevitas import config
from brevitas.core.stats.stats_op import AbsMax
from brevitas.core.stats.stats_op import AbsMinMax
from brevitas.core.stats.stats_op import NegativeMinOrZero
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantIdentity
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMSE
from brevitas.quant.float import Fp8e4m3Act
from brevitas.quant.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3FNUZDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3FNUZDynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPDynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat
from brevitas_examples.common.input_quantizer_builder import build_input_quantizer
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType

torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8
BIT_WIDTH = 8
# Matches the scaling_min_val carried by the reference activation quantizers.
SCALING_MIN_VAL = 1e-10

# Each spec reproduces an INPUT_QUANT_MAP['...']['static'] leaf through the
# generic InputQuantizerBuilder.
BUILDER_SPECS = {
    "int_static_per_tensor_sym": {
        "ref": Int8ActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_static_per_tensor_asym": {
        "ref": ShiftedUint8ActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_static_per_tensor_sym_mse": {
        "ref": Int8ActPerTensorFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            # The static-scaling mixin sets scaling_stats_op=PERCENTILE; the
            # builder's MSE scale init op would derive AbsPercentile from it,
            # while the reference MSESymmetricScale hardcodes AbsMax. Override
            # the init op to match.
            "kwargs": {
                "scaling_mse_init_op": AbsMax},},},
    "int_static_per_tensor_asym_mse": {
        "ref": ShiftedUint8ActPerTensorFloatMSE,
        # The asym MSE init ops (AbsMinMax / NegativeMinOrZero) take dtype/device
        # constructor args to build their `zero` buffer. ActQuantSolver provides
        # neither (only WeightQuantSolver does, via tracked_parameter_list), so
        # the MSE sub-injectors' `(this << 1).dtype/.device` cannot resolve.
        # Supplying dtype=None/device=None at the top level satisfies them (the
        # init ops default to None anyway). Applied to both the reference and the
        # builder so the two stay equivalent.
        "ref_kwargs": {
            "dtype": None, "device": None},
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "zero_point_param_method": ParamMethod.MSE,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            # The reference (MSEAsymmetricScale + MSEActZeroPoint) hardcodes the
            # scale init op to AbsMinMax and the zero-point init op to
            # NegativeMinOrZero; override the builder's derived ops to match.
            # dtype/device=None mirror the reference (see ref_kwargs above).
            "kwargs": {
                "scaling_mse_init_op": AbsMinMax,
                "zero_point_mse_init_op": NegativeMinOrZero,
                "dtype": None,
                "device": None},},},
    "int_static_per_tensor_sym_po2": {
        "ref": Int8ActPerTensorFixedPoint,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_static_per_tensor_sym_po2_mse": {
        "ref": Int8ActPerTensorFixedPointMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            # Same percentile-vs-AbsMax scale init op mismatch as the non-po2
            # symmetric MSE case; override to match the reference.
            "kwargs": {
                "scaling_mse_init_op": AbsMax},},},
    # ----------------------------------------------------------------------
    # float static (per_tensor sym): INPUT_QUANT_MAP['float'/'float_ocp'/
    # 'float_fnuz']['static']['float_scale']['stats']['per_tensor']['sym'].
    # ----------------------------------------------------------------------
    "float_static_per_tensor_sym": {
        "ref": Fp8e4m3ActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_ocp_static_per_tensor_sym": {
        "ref": Fp8e4m3OCPActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_fnuz_static_per_tensor_sym": {
        "ref": Fp8e4m3FNUZActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.STATIC,
            "float_format": FloatFormat.FNUZ,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    # ----------------------------------------------------------------------
    # dynamic (per_tensor): INPUT_QUANT_MAP['...']['dynamic']. Scale (and, for
    # asym, zero-point) are recomputed per-forward. Per-tensor granularity is
    # hosted directly by QuantIdentity; per_row/per_group need a group-aware
    # layer and are covered elsewhere.
    # ----------------------------------------------------------------------
    "int_dynamic_per_tensor_sym": {
        "ref": Int8DynamicActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_dynamic_per_tensor_asym": {
        "ref": ShiftedUint8DynamicActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_fnuz_dynamic_per_tensor_sym": {
        "ref": Fp8e4m3FNUZDynamicActPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.FNUZ,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    # ----------------------------------------------------------------------
    # dynamic (per_row / per_group): INPUT_QUANT_MAP['...']['dynamic'] with
    # per_row / per_group granularity. These cannot be hosted by a bare
    # QuantIdentity: per_row needs per_channel_broadcastable_shape and per_group
    # needs group_dim/group_size, both supplied by a QuantLinear input_quant. The
    # ``granularity`` key drives the harness, which also applies the same
    # generate_quantizers ``.let()`` overrides to both reference and builder.
    # ----------------------------------------------------------------------
    "int_dynamic_per_row_sym": {
        "ref": Int8DynamicActPerRowFloat,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_dynamic_per_row_asym": {
        "ref": ShiftedUint8DynamicActPerRowFloat,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    # dynamic po2 (per_row): fixed-point scale floors the exponent (FloorSte),
    # unlike static po2 activations which ceil it.
    "int_dynamic_per_row_sym_po2": {
        "ref": Int8DynamicActPerRowFixedPoint,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_ocp_dynamic_per_row_sym_po2": {
        "ref": FP8e4m3OCPDynamicActPerRowFixedPoint,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_ocp_dynamic_per_row_sym": {
        "ref": FP8e4m3OCPDynamicActPerRowFloat,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_fnuz_dynamic_per_row_sym": {
        "ref": FP8e4m3FNUZDynamicActPerRowFloat,
        "granularity": "per_row",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.FNUZ,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_dynamic_per_group_sym": {
        "ref": Int8DynamicActPerGroupFloat,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "int_dynamic_per_group_asym": {
        "ref": ShiftedUint8DynamicActPerGroupFloat,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        # The reference ShiftedUint8DynamicActPerGroupFloat crashes in its own
        # forward: the group zero-point (RuntimeDynamicGroupZeroPoint +
        # NegativeMinOrZero) does not reduce over the group-size dim, so the
        # zero-point stats (size group_size) fail to broadcast against the group
        # scale (size n_groups) in _ScaleShiftZeroPoint's x / scale. The builder
        # reproduces the reference structure exactly (module hierarchy matches),
        # so there is nothing correct to compare against.
        "xfail": "Reference groupwise asym dynamic activation crashes upstream.",},
    "float_dynamic_per_group_sym": {
        "ref": Fp8e4m3DynamicActPerGroupFloat,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_ocp_dynamic_per_group_sym": {
        "ref": Fp8e4m3OCPDynamicActPerGroupFloat,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    # ----------------------------------------------------------------------
    # no_scale (float, sym): INPUT_QUANT_MAP['float']['no_scale']. Float-only,
    # no scale at all (uses FloatActBase, constant unit scale). Per-tensor and
    # hosted directly by QuantIdentity.
    # ----------------------------------------------------------------------
    "float_no_scale_sym": {
        "ref": Fp8e4m3Act,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.NO_SCALE,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    # ----------------------------------------------------------------------
    # MX (dynamic po2 per_group): INPUT_QUANT_MAP['int'/'float_ocp']['dynamic']
    # ['po2_scale']['stats']['per_group']. Groupwise power-of-two dynamic scale;
    # float MX is OCP-only.
    # ----------------------------------------------------------------------
    "int_dynamic_per_group_sym_po2": {
        "ref": MXInt8Act,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},
    "float_ocp_dynamic_per_group_sym_po2": {
        "ref": MXFloat8e4m3Act,
        "granularity": "per_group",
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "scale_type": ScaleType.DYNAMIC,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},},}


def _make_quant_identity(act_quant, **kwargs):
    return QuantIdentity(act_quant=act_quant, return_quant_tensor=True, **kwargs)


def _module_hierarchy(model):
    hierarchy = []
    for name, module in model.named_modules():
        type_ = type(module)
        hierarchy.append((name, f"{type_.__module__}.{type_.__qualname__}"))
    return hierarchy


# generate_quantizers applies these runtime .let() overrides to the per_row /
# per_group dynamic activation quantizers (they are not baked into the reference
# classes). A bare QuantIdentity cannot auto-resolve the per-channel broadcastable
# shape (per_row) or group_dim (per_group), so we inject those manually too. The
# same overrides are applied to both the reference and the builder injectors so
# the comparison is fair.
def _apply_granularity_overrides(act_quant, granularity):
    if granularity == "per_row":
        # per_row scale is per output feature of a (N, IN_FEATURES) input.
        return act_quant.let(
            dynamic_scaling_broadcastable_fn=lambda x,
            shape: x.view(*shape[:-1], 1),
            permute_dims=None,
            stats_reduce_dim=1,
            per_channel_broadcastable_shape=(1, IN_FEATURES))
    if granularity == "per_group":
        return act_quant.let(group_dim=-1, group_size=GROUP_SIZE)
    return act_quant


def _run_and_compare(ref_out, builder_out):
    assert torch.equal(ref_out.value, builder_out.value)
    assert torch.equal(ref_out.scale, builder_out.scale)
    assert (ref_out.zero_point is None) == (builder_out.zero_point is None)
    if ref_out.zero_point is not None:
        assert torch.equal(ref_out.zero_point, builder_out.zero_point)
    if hasattr(ref_out, "bit_width"):
        assert torch.equal(ref_out.bit_width, builder_out.bit_width)
    else:
        assert torch.equal(ref_out.exponent_bit_width, builder_out.exponent_bit_width)
        assert torch.equal(ref_out.mantissa_bit_width, builder_out.mantissa_bit_width)


@pytest.mark.parametrize("spec_name", list(BUILDER_SPECS.keys()))
def test_builder_input_quant_matches_reference(spec_name):
    spec = BUILDER_SPECS[spec_name]
    ref_quant = spec["ref"]
    granularity = spec.get("granularity", "per_tensor")

    # Local-loss param methods (MSE, HQO) require JIT to be disabled.
    local_loss_methods = (ParamMethod.MSE, ParamMethod.HQO)
    param_methods = (
        spec["builder_args"].get("scaling_param_method"),
        spec["builder_args"].get("zero_point_param_method"))
    if config.JIT_ENABLED and any(m in local_loss_methods for m in param_methods):
        pytest.skip(reason="Local loss param methods (MSE, HQO) require JIT to be disabled")

    if "xfail" in spec:
        pytest.xfail(reason=spec["xfail"])

    builder_args = spec["builder_args"]
    # per_group builders need the group_size directly in the injector namespace.
    if granularity == "per_group":
        builder_args = {
            **builder_args, "kwargs": {
                **builder_args["kwargs"], "group_size": GROUP_SIZE}}
    builder_quant = build_input_quantizer(**builder_args).build_quant_injector()

    # All granularities are hosted by QuantIdentity; per_row / per_group inject the
    # otherwise layer-supplied attributes via .let() (see _apply_granularity_overrides).
    ref_quant = _apply_granularity_overrides(ref_quant, granularity)
    builder_quant = _apply_granularity_overrides(builder_quant, granularity)
    ref_act = _make_quant_identity(ref_quant, **spec.get("ref_kwargs", {}))
    builder_act = _make_quant_identity(builder_quant)

    # Module hierarchy must match 1-to-1.
    assert _module_hierarchy(ref_act) == _module_hierarchy(builder_act)

    # Collect identical runtime statistics on both, then compare the quantized
    # activations. Static scaling learns its scale from runtime stats, so we run
    # the same input through both in train mode before eval; dynamic scaling is
    # recomputed per-forward and is unaffected by the extra pass.
    x = torch.randn(8, IN_FEATURES)
    for act in (ref_act, builder_act):
        act.train()
        act(x)
        act.eval()
    _run_and_compare(ref_act(x), builder_act(x))
