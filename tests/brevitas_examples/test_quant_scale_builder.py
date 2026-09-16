# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Parity test for the quantized-scale weight builder.

Exercises the *generality* of the quantizer builder to a new quantizer family:
quantizers whose scale is itself quantized by a nested quantizer. The
:class:`QuantScaleWeightQuantizerBuilder` substitutes ``ScaleRestrictComponent``
with :class:`QuantScaleRestrictComponent`, which builds the nested scale quantizer
with the *same* builder framework and wires it through ``QuantRestrictValue``.

The builder output is compared against the reference
``QuantScaleMXFloat8e4m3Weight`` / ``QuantScaleMXFloat8e4m3WeightMSE`` (a groupwise
MX OCP float weight quantizer with a per-tensor OCP e4m3 quantized scale).

Requires ``BREVITAS_JIT=0`` for the MSE variant.
"""

import pytest
import torch

from brevitas import config
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantLinear
from brevitas_examples.common.generative.quantizers import QuantScaleMXFloat8e4m3Weight
from brevitas_examples.common.generative.quantizers import QuantScaleMXFloat8e4m3WeightMSE
from brevitas_examples.common.quantizer_builder import default_scale_quantizer_config
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import FloatFormatConfig
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import QuantScaleQuantizerConfig
from brevitas_examples.common.quantizer_builder import QuantScaleWeightQuantizerBuilder

torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8
BIT_WIDTH = 8
# QuantLinear weight is (out, in); the weight solver auto-derives group_dim=1 for a
# non-transposed layer, so we pass the same value to satisfy the config's groupwise
# requirement without diverging from the reference.
GROUP_DIM = 1

# Groupwise MX OCP float weight, scale quantized to per-tensor OCP e4m3.
BUILDER_SPECS = {
    "quant_scale_mx_float_per_group_sym": {
        "ref": QuantScaleMXFloat8e4m3Weight,
        "scaling_impl_type": ScalingImplType.STATS,
        "scaling_param_method": ParamMethod.STATS,},
    "quant_scale_mx_float_per_group_sym_mse": {
        "ref": QuantScaleMXFloat8e4m3WeightMSE,
        "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
        "scaling_param_method": ParamMethod.MSE,},}


def _make_outer_config(spec) -> QuantScaleQuantizerConfig:
    """Groupwise MX OCP-float weight config whose scale is itself quantized (the
    nested scale config is passed via ``scale_config``, i.e. through ``build``)."""
    return QuantScaleQuantizerConfig(
        format=FloatFormatConfig(float_quant_format="e4m3", float_format=FloatFormat.OCP),
        quant_param_type=QuantParamType.SYM,
        scaling_granularity=ScalingPerOutputType.GROUP,
        scaling_impl_type=spec["scaling_impl_type"],
        restrict_scaling_type=RestrictValueType.QUANT,
        scaling_param_method=spec["scaling_param_method"],
        extra={
            "group_size": GROUP_SIZE, "group_dim": GROUP_DIM},
        scale_config=default_scale_quantizer_config())


def _make_quant_linear(weight_quant):
    return QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=weight_quant,
        return_quant_tensor=False,
        weight_group_size=GROUP_SIZE)


def _module_hierarchy(model):
    return [
        (name, f"{type(m).__module__}.{type(m).__qualname__}") for name, m in model.named_modules()]


@pytest.mark.parametrize("spec_name", list(BUILDER_SPECS.keys()))
def test_builder_quant_scale_weight_matches_reference(spec_name):
    spec = BUILDER_SPECS[spec_name]

    if config.JIT_ENABLED and spec["scaling_param_method"] == ParamMethod.MSE:
        pytest.skip(reason="Local loss param methods (MSE) require JIT to be disabled")

    ref_linear = _make_quant_linear(spec["ref"])

    builder = QuantScaleWeightQuantizerBuilder(_make_outer_config(spec))
    builder_linear = _make_quant_linear(builder.build_quant_injector())

    # 1) Module hierarchy must match 1-to-1 (structural parity of the injector,
    # including the nested scale quantizer).
    assert _module_hierarchy(ref_linear) == _module_hierarchy(builder_linear)

    # 2) Identical float weights so only the quantization path can differ.
    builder_linear.weight.data.copy_(ref_linear.weight.data)
    ref_linear.eval()
    builder_linear.eval()

    # Forward to trigger lazy scale initialization.
    mock_input = torch.randn(1, IN_FEATURES)
    ref_linear(mock_input)
    builder_linear(mock_input)

    # 3) Quantized weight tensors must match exactly.
    ref_weight = ref_linear.quant_weight()
    builder_weight = builder_linear.quant_weight()
    assert torch.equal(ref_weight.value, builder_weight.value)
    assert torch.equal(ref_weight.scale, builder_weight.scale)
    assert torch.equal(ref_weight.exponent_bit_width, builder_weight.exponent_bit_width)
    assert torch.equal(ref_weight.mantissa_bit_width, builder_weight.mantissa_bit_width)

    # 4) Quantized layer outputs (the full forward) must match exactly.
    x = torch.randn(4, IN_FEATURES)
    assert torch.equal(ref_linear(x), builder_linear(x))
