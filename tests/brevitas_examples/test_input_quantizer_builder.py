# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the input/activation quantizer builder.

Covers the ``static`` and ``dynamic`` scale types of ``INPUT_QUANT_MAP`` (the
``no_scale`` scale type is not implemented in the builder yet).
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
from brevitas.quant.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3FNUZDynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat
from brevitas_examples.common.quantizer_builder import build_input_quantizer
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType

torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
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
            "kwargs": {},},},}


def _make_quant_identity(act_quant, **kwargs):
    return QuantIdentity(act_quant=act_quant, return_quant_tensor=True, **kwargs)


def _module_hierarchy(model):
    hierarchy = []
    for name, module in model.named_modules():
        type_ = type(module)
        hierarchy.append((name, f"{type_.__module__}.{type_.__qualname__}"))
    return hierarchy


@pytest.mark.parametrize("spec_name", list(BUILDER_SPECS.keys()))
def test_builder_input_quant_matches_reference(spec_name):
    spec = BUILDER_SPECS[spec_name]
    ref_quant = spec["ref"]

    # Local-loss param methods (MSE, HQO) require JIT to be disabled.
    local_loss_methods = (ParamMethod.MSE, ParamMethod.HQO)
    param_methods = (
        spec["builder_args"].get("scaling_param_method"),
        spec["builder_args"].get("zero_point_param_method"))
    if config.JIT_ENABLED and any(m in local_loss_methods for m in param_methods):
        pytest.skip(reason="Local loss param methods (MSE, HQO) require JIT to be disabled")

    if "xfail" in spec:
        pytest.xfail(reason=spec["xfail"])

    ref_act = _make_quant_identity(ref_quant, **spec.get("ref_kwargs", {}))
    builder = build_input_quantizer(**spec["builder_args"])
    builder_act = _make_quant_identity(builder.build_quant_injector())

    # 1) Module hierarchy must match 1-to-1.
    assert _module_hierarchy(ref_act) == _module_hierarchy(builder_act)

    # 2) Collect identical runtime statistics on both, then compare the
    # quantized activations. Static scaling learns its scale from runtime stats,
    # so we run the same input through both in train mode before eval.
    x = torch.randn(8, IN_FEATURES)
    ref_act.train()
    builder_act.train()
    ref_act(x)
    builder_act(x)

    ref_act.eval()
    builder_act.eval()
    ref_out = ref_act(x)
    builder_out = builder_act(x)

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
