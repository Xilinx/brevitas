"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Weight quantizer builder (v2): the concrete :class:`WeightQuantizerBuilder` with
its ordered component list. Instantiate it via the shared
``build_quantizer(WeightQuantizerBuilder, ...)`` factory (see :mod:`.builder`).
"""
from typing import List

from dependencies import this

from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.generative.quant_blocks import QuantScaleScaleShapeMixin
from brevitas_examples.common.quantizer_builder.builder import QuantizerBuilder
from brevitas_examples.common.quantizer_builder.components import BaseComponent
from brevitas_examples.common.quantizer_builder.components import FormatComponent
from brevitas_examples.common.quantizer_builder.components import QuantScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.components import ScaleComponent
from brevitas_examples.common.quantizer_builder.components import ScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.components import WeightSolverComponent
from brevitas_examples.common.quantizer_builder.components import ZeroPointComponent
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import FloatFormatConfig
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType


class WeightQuantizerBuilder(QuantizerBuilder):
    """Builds a weight quantizer injector.

    The ordered component list ends with the weight-specific solver and int
    tuning: the solver contributes the lowest-priority base (matching the
    reference weight quantizers' MRO) and the int tuning has the final say over
    the signed / narrow-range attributes.
    """

    def base_components(self) -> List[Component]:
        return [
            ScaleComponent(),
            ZeroPointComponent(),
            FormatComponent(),
            ScaleRestrictComponent(),
            BaseComponent(),
            WeightSolverComponent(),]


def default_scale_quantizer_config() -> QuantizerConfig:
    """Config for the nested quantizer that quantizes the scale: a per-tensor OCP
    e4m3 float weight quantizer (matches the reference ``QuantWeightScalingFloat``
    base ``Fp8e4m3OCPWeightPerTensorFloat``).

    The ``this << 1`` parent references (module / tracked parameters / upstream
    granularity) that the nested scale quantizer reads from its enclosing
    quantizer are carried as ``extra`` (injector namespace attributes), so they
    flow through the builder like any other namespace attribute rather than being
    layered on afterwards.
    """
    return QuantizerConfig(
        format=FloatFormatConfig(float_quant_format="e4m3", float_format=FloatFormat.OCP),
        quant_param_type=QuantParamType.SYM,
        scaling_granularity=ScalingPerOutputType.TENSOR,
        scaling_impl_type=ScalingImplType.STATS,
        restrict_scaling_type=RestrictValueType.FP,
        scaling_param_method=ParamMethod.STATS,
        extra={
            "module": (this << 1).module,
            "tracked_parameter_list": (this << 1).tracked_parameter_list,
            "upstream_scaling": (this << 1).scaling_per_output_type,},
        # The quant-scale shape mixin is folded in as an extra base (last in the
        # MRO), so the nested builder produces the complete scale injector.
        extra_bases=(QuantScaleScaleShapeMixin,))


class QuantScaleWeightQuantizerBuilder(WeightQuantizerBuilder):
    """Weight builder whose scale is itself quantized.

    Substitutes :class:`ScaleRestrictComponent` with the stateless
    :class:`QuantScaleRestrictComponent`, which reads the nested scale config from
    the (:class:`~.core.QuantScaleQuantizerConfig`) ``config`` passed to
    ``build``. Reproduces the reference ``QuantScaleMXFloat8e4m3Weight`` when the
    outer config is a groupwise (MX) OCP float quantizer with
    ``restrict_scaling_type == RestrictValueType.QUANT``.
    """

    def base_components(self) -> List[Component]:
        return [
            ScaleComponent(),
            ZeroPointComponent(),
            FormatComponent(),
            QuantScaleRestrictComponent(),
            BaseComponent(),
            WeightSolverComponent(),]
