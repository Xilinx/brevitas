"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Concrete :class:`Component` implementations for the quantizer builder.

Each component reads what it needs from the :class:`QuantizerConfig` (Context
Object) and returns a :class:`Contribution` (namespace attrs + base mixins). The
:class:`QuantizerBuilder` folds an ordered list of these into a brevitas
injector. This module holds the kind-agnostic (shared) components as well as the
kind-specific (weight / input) ones.
"""
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant.float_base import ScaledFloatWeightBase
from brevitas.quant.solver.common import solve_float_to_int_impl_from_enum
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas_examples.common.quant_builder_core import Component
from brevitas_examples.common.quant_builder_core import Contribution
from brevitas_examples.common.quant_builder_core import QuantizerConfig
from brevitas_examples.common.quantizer_builder import AsymmetricZeroPointMixin
from brevitas_examples.common.quantizer_builder import FLOAT_FORMAT_MIXIN_MAP
from brevitas_examples.common.quantizer_builder import GroupwisePoTMixin
from brevitas_examples.common.quantizer_builder import HQOScaleInjectorMixin
from brevitas_examples.common.quantizer_builder import HQOZeroPointInjectorMixin
from brevitas_examples.common.quantizer_builder import MSEScaleInjectorMixin
from brevitas_examples.common.quantizer_builder import MSEZeroPointInjectorMixin
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import parse_float_quant_format
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import QuantType


class CommonComponent(Component):
    """Kind-agnostic namespace attributes shared by every quantizer.

    ``bit_width_impl_type`` / ``float_to_int_impl_type`` are constant across all
    supported quantizers (no reference varies them). ``scaling_impl_type`` is
    intentionally not set here: it is kind-specific (weight -> STATS; input ->
    derived from the scale type) and set by the kind/scale components.
    """

    def build(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "bit_width_impl_type": BitWidthImplType.CONST,
                "float_to_int_impl_type": FloatToIntImplType.ROUND,
                "scaling_per_output_type": config.scaling_granularity,
                "restrict_scaling_type": config.restrict_scaling_type,
                "scaling_min_val": config.scaling_min_val,})


class FormatComponent(Component):
    """Int vs float namespace attributes, plus the float format mixin (OCP/FNUZ)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return self.build_int(config) if config.is_int else self.build_float(config)

    def build_int(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "quant_type": QuantType.INT,
                "bit_width": config.format.bit_width,})

    def build_float(self, config: QuantizerConfig) -> Contribution:
        exponent_bit_width, mantissa_bit_width, bit_width = parse_float_quant_format(
            config.format.float_quant_format)
        format_mixin = FLOAT_FORMAT_MIXIN_MAP.get(config.format.float_format.value)
        bases = (format_mixin,) if format_mixin is not None else ()
        return Contribution(
            attrs={
                "quant_type": QuantType.FP,
                "bit_width": bit_width,
                "exponent_bit_width": exponent_bit_width,
                "mantissa_bit_width": mantissa_bit_width,
                # All FloatFormat mixins set saturating=True; set it for the plain
                # FLOAT format too (the reference Fp8e4m3Mixin sets it).
                "saturating": True,},
            bases=bases)


# TODO (pml): Refactor to avoid duplication with ScaleParamMethodComponent
class ScaleComponent(Component):
    """Base scale wiring (counterpart of :class:`ZeroPointComponent`). Sets the
    scale implementation type (default STATS from the config). MSE/HQO drop it and
    the input scale extra overrides it for the static/dynamic/no_scale paths."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return Contribution(attrs={"scaling_impl_type": config.scaling_impl_type})


class ScaleParamMethodComponent(Component):
    """Scale parameter method: MSE / HQO local-loss injectors (STATS = nothing).

    MSE/HQO force a parameter-from-stats scale, so any ``scaling_impl_type`` set
    by an earlier component is dropped.
    """

    def build(self, config: QuantizerConfig) -> Contribution:
        if config.scaling_param_method == ParamMethod.MSE:
            return Contribution(bases=(MSEScaleInjectorMixin,), drop=("scaling_impl_type",))
        if config.scaling_param_method == ParamMethod.HQO:
            return Contribution(bases=(HQOScaleInjectorMixin,), drop=("scaling_impl_type",))
        return Contribution()


# TODO (pml): Refactor to avoid duplication with ScaleParamMethodComponent
class ZeroPointParamMethodComponent(Component):
    """Zero-point parameter method: MSE / HQO local-loss injectors (only relevant
    for asymmetric quantizers; None = nothing)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        if config.zero_point_param_method == ParamMethod.MSE:
            return Contribution(bases=(MSEZeroPointInjectorMixin,))
        if config.zero_point_param_method == ParamMethod.HQO:
            return Contribution(bases=(HQOZeroPointInjectorMixin,))
        return Contribution()


class ScaleRestrictComponent(Component):
    """Power-of-two *scale* handling: rounding of the exponent + the groupwise (MX)
    mixin. Non-group po2 ceils the exponent by default; groupwise (MX) floors it.
    Kind-specific rounding overrides (e.g. input dynamic po2 -> floor) are applied
    by a kind component. This restricts only the scale; zero-point restriction is
    handled elsewhere.
    """

    def build(self, config: QuantizerConfig) -> Contribution:
        if config.restrict_scaling_type != RestrictValueType.POWER_OF_TWO:
            return Contribution()
        if config.scaling_granularity == ScalingPerOutputType.GROUP:
            return self.build_po2_restrict_groupwise(config)
        return self.build_po2_restrict_non_groupwise(config)

    def build_po2_restrict_groupwise(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "restrict_value_float_to_int_impl":
                    solve_float_to_int_impl_from_enum(FloatToIntImplType.FLOOR)},
            bases=(GroupwisePoTMixin,))

    def build_po2_restrict_non_groupwise(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "restrict_value_float_to_int_impl":
                    solve_float_to_int_impl_from_enum(FloatToIntImplType.CEIL)})


class ZeroPointComponent(Component):
    """Wires the zero point. Symmetric quantizers get a fixed zero zero-point (and
    the max-abs scale stats op); asymmetric quantizers get the stats-based
    asymmetric zero-point mixin. Kind-specific tuning (zero_point_impl_type,
    stats ops, runtime/dynamic zero-point) is applied by kind components."""

    def build(self, config: QuantizerConfig) -> Contribution:
        is_asym = QuantParamType(config.quant_param_type) == QuantParamType.ASYM
        return self.build_asym(config) if is_asym else self.build_sym(config)

    def build_sym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "zero_point_impl": ZeroZeroPoint, "scaling_stats_op": StatsOp.MAX})

    def build_asym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(bases=(AsymmetricZeroPointMixin,))


class WeightSolverComponent(Component):
    """Weight solver, float base and proxy class. The scale implementation type is
    provided by the base :class:`ScaleComponent` (default STATS)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return self.build_int(config) if config.is_int else self.build_float(config)

    def build_int(self, config: QuantizerConfig) -> Contribution:
        groupwise = config.scaling_granularity == ScalingPerOutputType.GROUP
        proxy = GroupwiseWeightQuantProxyFromInjector if groupwise \
            else WeightQuantProxyFromInjector
        return Contribution(attrs={"proxy_class": proxy}, bases=(WeightQuantSolver,))

    def build_float(self, config: QuantizerConfig) -> Contribution:
        # ScaledFloatWeightBase already brings the weight solver and a stats scale.
        groupwise = config.scaling_granularity == ScalingPerOutputType.GROUP
        proxy = GroupwiseWeightFloatQuantProxyFromInjector if groupwise \
            else WeightFloatQuantProxyFromInjector
        return Contribution(attrs={"proxy_class": proxy}, bases=(ScaledFloatWeightBase,))


class WeightIntQuantComponent(Component):
    """Signedness / narrow-range / zero-point enable for integer weights (no-op
    for float, whose signedness comes from the float base)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        if not config.is_int:
            return Contribution()
        is_asym = QuantParamType(config.quant_param_type) == QuantParamType.ASYM
        return self.build_asym(config) if is_asym else self.build_sym(config)

    def build_sym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(attrs={"signed": True, "narrow_range": True})

    def build_asym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "signed": False, "narrow_range": False, "quantize_zero_point": True})
