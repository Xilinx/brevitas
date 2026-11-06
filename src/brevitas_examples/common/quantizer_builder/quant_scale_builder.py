"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Quantized-scale support: quantizers whose scale is itself quantized by a nested
quantizer. Groups the config (:class:`QuantScaleQuantizerConfig`), the injector
mixin (:class:`QuantScaleMixin`), the component that substitutes the plain
scale restriction (:class:`QuantScaleRestrictComponent`) and the weight builder
that uses it (:class:`QuantScaleWeightQuantizerBuilder`). Mirrors the reference
``QuantScaleMXFloat8e4m3Weight``.
"""
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Type

from dependencies import this
from dependencies import value

from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.generative.quant_blocks import QuantScaleScaleShapeMixin
from brevitas_examples.common.quantizer_builder.components import BaseComponent
from brevitas_examples.common.quantizer_builder.components import FormatComponent
from brevitas_examples.common.quantizer_builder.components import ScaleComponent
from brevitas_examples.common.quantizer_builder.components import ScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.components import WeightSolverComponent
from brevitas_examples.common.quantizer_builder.components import ZeroPointComponent
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import FloatFormatConfig
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType
from brevitas_examples.common.quantizer_builder.weight import WeightQuantizerBuilder


class QuantScaleMixin(ExtendedInjector):
    """Restrict the scale through a *nested* quantizer (a quantized scale).

    Replaces the power-of-two / float scale restriction: instead of rounding the
    scale to a grid, it is quantized by ``scaling_float_quant`` (a nested quantizer
    injector supplied by :class:`QuantScaleRestrictComponent`).
    Mirrors the reference ``QuantScaleMXFloat8e4m3Weight``.
    """
    restrict_scaling_impl = QuantRestrictValue
    restrict_threshold_impl = FloatRestrictValue
    restrict_threshold_with_scale = True

    @value
    def restrict_value_float_to_int_impl():
        # The scale is "rounded" by quantizing it through the nested scale quantizer.
        return this.scaling_float_quant.tensor_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        # Only groupwise scales are reshaped back to their (expanded) scaling shape
        # after de-quantization; per-tensor / per-channel keep None.
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape
        return None


@dataclass(frozen=True)
class QuantScaleQuantizerConfig(QuantizerConfig):
    """A :class:`QuantizerConfig` whose scale is itself quantized by a *nested*
    quantizer, described by ``scale_config``.

    Used together with ``restrict_scaling_type == RestrictValueType.QUANT``. It is
    passed to the builder like any other config, so the quant-scale component reads
    the nested config from ``build(config)`` without holding any state.
    """
    scale_config: Optional[QuantizerConfig] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.restrict_scaling_type == RestrictValueType.QUANT and self.scale_config is None:
            raise ValueError(
                "QuantScaleQuantizerConfig requires a `scale_config` when "
                "`restrict_scaling_type == RestrictValueType.QUANT`.")


class QuantScaleRestrictComponent(ScaleRestrictComponent):
    """Quantized-scale *scale* handling: substitutes :class:`ScaleRestrictComponent`.

    When the config opts into ``RestrictValueType.QUANT`` it reads the
    nested scale config from ``config.scale_config`` (a
    :class:`QuantScaleQuantizerConfig`) and quantizes the scale with that
    nested quantizer (``scaling_float_quant``) instead of rounding it to a power of
    two. The restrict wiring comes from :class:`QuantScaleMixin`. Mirrors the
    reference ``QuantScaleMXFloat8e4m3Weight``. Any other ``restrict_scaling_type``
    falls back to the plain :class:`ScaleRestrictComponent` behaviour.
    """

    def build(self, config: QuantizerConfig) -> Contribution:
        if config.restrict_scaling_type != RestrictValueType.QUANT:
            return super().build(config)
        if not hasattr(config, "scale_config") or config.scale_config is None:
            raise ValueError(
                "RestrictValueType.QUANT requires a QuantScaleQuantizerConfig with a `scale_config`."
            )
        return Contribution(
            attrs={
                "restrict_scaling_type": config.restrict_scaling_type,
                "scaling_float_quant": self._build_inner_scale_injector(config.scale_config),},
            bases=(QuantScaleMixin,))

    def _build_inner_scale_injector(self, config: QuantizerConfig) -> Type:
        # The nested builder produces the complete scale injector: the ``this << 1``
        # upstream references and the quant-scale shape mixin (last in the MRO) are
        # carried by the scale config's ``attr_overrides`` / ``base_overrides``.
        return WeightQuantizerBuilder(config).build_quant_injector()


def create_base_scale_quantizer_config() -> QuantizerConfig:
    """Config for the nested quantizer that quantizes the scale: a per-tensor OCP
    e4m3 float weight quantizer (matches the reference ``QuantWeightScalingFloat``
    base ``Fp8e4m3OCPWeightPerTensorFloat``).

    The ``this << 1`` parent references (module / tracked parameters / upstream
    granularity) that the nested scale quantizer reads from its enclosing
    quantizer are carried as ``attr_overrides`` (injector namespace attributes), so
    they flow through the builder like any other namespace attribute rather than
    being layered on afterwards.
    """
    return QuantizerConfig(
        format=FloatFormatConfig(float_quant_format="e4m3", float_format=FloatFormat.OCP),
        quant_param_type=QuantParamType.SYM,
        scaling_granularity=ScalingPerOutputType.TENSOR,
        scaling_impl_type=ScalingImplType.STATS,
        restrict_scaling_type=RestrictValueType.FP,
        scaling_param_method=ParamMethod.STATS,
        attr_overrides={
            "module": (this << 1).module,
            "tracked_parameter_list": (this << 1).tracked_parameter_list,
            "upstream_scaling": (this << 1).scaling_per_output_type,},
        # The quant-scale shape mixin is folded in as an extra base (last in the
        # MRO), so the nested builder produces the complete scale injector.
        base_overrides=(QuantScaleScaleShapeMixin,))


class QuantScaleWeightQuantizerBuilder(WeightQuantizerBuilder):
    """Weight builder whose scale is itself quantized.

    Substitutes :class:`ScaleRestrictComponent` with the stateless
    :class:`QuantScaleRestrictComponent`, which reads the nested scale config from
    the (:class:`QuantScaleQuantizerConfig`) ``config`` passed to
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
