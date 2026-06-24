# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.core.stats import NegativeMinOrZero
from brevitas.core.zero_point import StatsFromParameterZeroPoint
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import this
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.nn import QuantLinear
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.common.quantizer_builder import QuantizerBuilder

# Keep the model small and deterministic so that weight-quant outputs are
# directly comparable between the reference quantizer and the builder.
torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8
BIT_WIDTH = 8
# Matches the ``scaling_min_val`` carried by the reference weight quantizers
# through MaxStatsScaling / MinMaxStatsScaling (brevitas.quant.base).
SCALING_MIN_VAL = 1e-10


# A builder spec describes how to reproduce a given reference quantizer class
# from WEIGHT_QUANT_MAP['int']['float_scale']['stats'] through the generic
# QuantizerBuilder. Each spec carries:
#   - ``ref``: the reference quantizer class (the WEIGHT_QUANT_MAP leaf)
#   - ``builder_args``: kwargs passed to QuantizerBuilder.__init__
#   - ``layer_kwargs``: extra kwargs for the QuantLinear wrapping the quantizer
#     (e.g. group_size for groupwise quantization)
#
# The ``kwargs`` entry of ``builder_args`` carries the directives that are not
# (yet) first-class builder parameters but are required to match the reference
# quantizer: signedness, narrow range, zero-point handling and proxy class.
def _sym_kwargs():
    return {
        "signed": True,
        "narrow_range": True,
        "zero_point_impl": ZeroZeroPoint,}


def _asym_kwargs():
    return {
        "signed": False,
        "narrow_range": False,
        "quantize_zero_point": True,
        "zero_point_impl": StatsFromParameterZeroPoint,
        "zero_point_stats_impl": NegativeMinOrZero,
        "zero_point_shape": this.scaling_shape,
        "zero_point_stats_input_view_shape_impl": this.scaling_stats_input_view_shape_impl,
        "zero_point_stats_input_concat_dim": this.scaling_stats_input_concat_dim,}


BUILDER_SPECS = {
    "int_per_tensor_sym": {
        "ref": Int8WeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MAX,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_sym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},
    "int_per_tensor_asym": {
        "ref": ShiftedUint8WeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MIN_MAX,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_asym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},
    "int_per_channel_sym": {
        "ref": Int8WeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MAX,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_sym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},
    "int_per_channel_asym": {
        "ref": ShiftedUint8WeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MIN_MAX,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_asym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},
    "int_per_group_sym": {
        "ref": IntWeightSymmetricGroupQuant,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MAX,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_sym_kwargs(),
                "group_size": GROUP_SIZE,
                "proxy_class": GroupwiseWeightQuantProxyFromInjector,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "int_per_group_asym": {
        "ref": ShiftedUint8WeightGroupQuantFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_stats_op": StatsOp.MIN_MAX,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_asym_kwargs(),
                "group_size": GROUP_SIZE,
                "proxy_class": GroupwiseWeightQuantProxyFromInjector,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    # ----------------------------------------------------------------------
    # MSE param method: WEIGHT_QUANT_MAP['int']['float_scale']['mse'].
    # The scale is learned/initialized from an MSE local loss, selected on the
    # builder side via scaling_param_method="mse".
    #
    # NOTE: Only the *symmetric* MSE quantizers are covered here. The asymmetric
    # MSE quantizers (ShiftedUint8Weight*FloatMSE) also learn the zero-point
    # from an MSE local loss (via MSEWeightZeroPoint), which the builder does
    # not generate yet. TODO: add per_tensor/per_channel asym MSE specs once the
    # builder supports MSE-based zero-points.
    # ----------------------------------------------------------------------
    "int_per_tensor_sym_mse": {
        "ref": Int8WeightPerTensorFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": "mse",
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_stats_op": StatsOp.MAX,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_sym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},
    "int_per_channel_sym_mse": {
        "ref": Int8WeightPerChannelFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": "mse",
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_stats_op": StatsOp.MAX,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                **_sym_kwargs(),
                "proxy_class": WeightQuantProxyFromInjector,},},
        "layer_kwargs": {},},}


def _make_quant_linear(weight_quant, **layer_kwargs):
    # NOTE: return_quant_tensor must be False here. With weight-only
    # quantization (no input_quant), the layer input stays a plain Tensor, so
    # the layer cannot emit a QuantTensor output and would otherwise raise
    # "QuantLayer is not correctly configured". The quantized weight is still
    # compared directly via ``quant_weight()``, which returns a QuantTensor
    # regardless of this flag.
    linear = QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=weight_quant,
        return_quant_tensor=False,
        **layer_kwargs)
    return linear


def _module_hierarchy(model):
    """Return an ordered, comparable description of the module hierarchy.

    Each entry is a (name, fully-qualified-type) pair, so two models match
    1-to-1 only if they have exactly the same submodules, in the same order,
    of the same types.
    """
    hierarchy = []
    for name, module in model.named_modules():
        type_ = type(module)
        hierarchy.append((name, f"{type_.__module__}.{type_.__qualname__}"))
    return hierarchy


@pytest.mark.parametrize("spec_name", list(BUILDER_SPECS.keys()))
def test_builder_weight_quant_matches_reference(spec_name):
    spec = BUILDER_SPECS[spec_name]
    ref_quant = spec["ref"]
    layer_kwargs = spec["layer_kwargs"]

    # Reference layer built directly from the WEIGHT_QUANT_MAP leaf class.
    ref_linear = _make_quant_linear(ref_quant, **layer_kwargs)

    # Builder layer built from the generic QuantizerBuilder.
    builder = QuantizerBuilder(**spec["builder_args"])
    builder_quant = builder.build_quant_injector()
    builder_linear = _make_quant_linear(builder_quant, **layer_kwargs)

    # 1) Module hierarchy must match 1-to-1. Checked before syncing weights so a
    # structural mismatch is reported as a clear hierarchy diff rather than an
    # opaque "Missing key(s) in state_dict" error.
    assert _module_hierarchy(ref_linear) == _module_hierarchy(builder_linear)

    # Make both layers carry identical float weights so the only difference that
    # could appear is in the quantization path itself. We copy only the float
    # weight (not the full state_dict): for MSE / PARAMETER_FROM_STATS scaling
    # the learned scale parameter (scaling_impl.value) is excluded from
    # state_dict() until it has been initialized, so a strict load_state_dict
    # would spuriously fail with a missing key.
    builder_linear.weight.data.copy_(ref_linear.weight.data)

    ref_linear.eval()
    builder_linear.eval()

    # Mock forward pass to trigger lazy initialization of any parameter-based
    # scaling (e.g. MSE / PARAMETER_FROM_STATS). After this, scaling_impl.value
    # is initialized from the (now identical) weights on both layers, so the
    # learned scales are directly comparable.
    mock_input = torch.randn(1, IN_FEATURES)
    ref_linear(mock_input)
    builder_linear(mock_input)

    # 2) The quantized weight tensors themselves must match exactly.
    ref_weight = ref_linear.quant_weight()
    builder_weight = builder_linear.quant_weight()
    assert torch.equal(ref_weight.value, builder_weight.value)
    assert torch.equal(ref_weight.scale, builder_weight.scale)
    assert (ref_weight.zero_point is None) == (builder_weight.zero_point is None)
    if ref_weight.zero_point is not None:
        assert torch.equal(ref_weight.zero_point, builder_weight.zero_point)
    assert torch.equal(ref_weight.bit_width, builder_weight.bit_width)

    # 3) Quantized layer output tensors must match exactly. With
    # return_quant_tensor=False the layers return plain Tensors.
    x = torch.randn(4, IN_FEATURES)
    ref_out = ref_linear(x)
    builder_out = builder_linear(x)
    assert torch.equal(ref_out, builder_out)
