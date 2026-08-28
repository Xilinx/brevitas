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
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Literal
from typing import Type

from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.function_wrapper.shape import OverTensorView
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import RuntimeDynamicGroupZeroPoint
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.quant.float_base import FloatActBase
from brevitas.quant.float_base import ScaledFloatActBase
from brevitas.quant.float_base import ScaledFloatWeightBase
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.common import solve_float_to_int_impl_from_enum
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas_examples.common.generative.quant_blocks import RuntimeDynamicStatsZeroPoint
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.mixins import AsymmetricZeroPointMixin
from brevitas_examples.common.quantizer_builder.mixins import FLOAT_FORMAT_MIXIN_MAP
from brevitas_examples.common.quantizer_builder.mixins import GroupwisePoTMixin
from brevitas_examples.common.quantizer_builder.mixins import HQOScaleInjectorMixin
from brevitas_examples.common.quantizer_builder.mixins import HQOZeroPointInjectorMixin
from brevitas_examples.common.quantizer_builder.mixins import MSEScaleInjectorMixin
from brevitas_examples.common.quantizer_builder.mixins import MSEZeroPointInjectorMixin
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import parse_float_quant_format
from brevitas_examples.common.quantizer_builder.mixins import ZeroPointImplType


def _sym_scaling_stats_op(config: QuantizerConfig, default: StatsOp) -> StatsOp:
    """Symmetric scale-stats op, upgraded to the signed variant for a signed
    (SIGNED_FP) scale. Signed scales are a symmetric-only concept, so this helper
    is used exclusively on the symmetric scale paths; asymmetric quantizers keep
    their MIN_MAX stats op regardless of the restrict-value type."""
    return StatsOp.SIGNED_MAX if config.is_signed_scale else default


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


class ScaleComponent(Component):
    """Generic scale wiring (counterpart of :class:`ZeroPointComponent`): sets the
    scale implementation type from the config (the act / weight solver resolves it
    to a concrete scaling impl). :class:`InputScaleComponent` subclasses this to add
    the activation-only static / dynamic / no_scale overrides."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return Contribution(attrs={"scaling_impl_type": config.scaling_impl_type})


class ParamMethodComponent(Component):
    """Template: select the MSE / HQO local-loss mixin for a target (scale or
    zero-point). Subclasses supply the config field, the target name (for the error
    message) and the MSE / HQO mixin pair. STATS / None contributes nothing."""

    param_method_attr: Literal["scaling_param_method", "zero_point_param_method"]
    target: Literal["scale", "zero-point"]
    mse_mixin: Type
    hqo_mixin: Type

    def validate(self, config: QuantizerConfig) -> None:
        # MSE/HQO calibrate a stored parameter once; a dynamic scale is recomputed
        # per-forward, so the two are mutually exclusive.
        if getattr(config, self.param_method_attr) in (ParamMethod.MSE,
                                                       ParamMethod.HQO) and config.is_dynamic:
            raise ValueError(f"MSE/HQO {self.target} is incompatible with a dynamic scale.")

    def build(self, config: QuantizerConfig) -> Contribution:
        match getattr(config, self.param_method_attr):
            case ParamMethod.MSE:
                return Contribution(bases=(self.mse_mixin,))
            case ParamMethod.HQO:
                return Contribution(bases=(self.hqo_mixin,))
            case _:
                return Contribution()


class ScaleParamMethodComponent(ParamMethodComponent):
    """Scale parameter method: MSE / HQO local-loss injectors (STATS = nothing)."""

    param_method_attr = "scaling_param_method"
    target = "scale"
    mse_mixin = MSEScaleInjectorMixin
    hqo_mixin = HQOScaleInjectorMixin


class ZeroPointParamMethodComponent(ParamMethodComponent):
    """Zero-point parameter method: MSE / HQO local-loss injectors (only relevant
    for asymmetric quantizers; None = nothing)."""

    param_method_attr = "zero_point_param_method"
    target = "zero-point"
    mse_mixin = MSEZeroPointInjectorMixin
    hqo_mixin = HQOZeroPointInjectorMixin


class ScaleRestrictComponent(Component):
    """Power-of-two *scale* handling: rounding of the exponent + the groupwise (MX)
    mixin. Groupwise (MX) floors the exponent. Non-group po2 ceils it for static
    scales but floors it for dynamic scales (mirrors brevitas
    Int8DynamicActPerRowFixedPoint / FP8e4m3OCPDynamicActPerRowFixedPoint). This
    restricts only the scale; zero-point restriction is handled elsewhere.
    """

    def build(self, config: QuantizerConfig) -> Contribution:
        if not config.is_power_of_two:
            return Contribution()
        # Groupwise (MX) and dynamic (activation) po2 scales floor the exponent;
        # non-group static scales ceil it (weights never use dynamic scaling). The
        # groupwise (MX) mixin is added only for groupwise.
        rounding = FloatToIntImplType.FLOOR if (config.is_groupwise or config.is_dynamic) \
            else FloatToIntImplType.CEIL
        bases = (GroupwisePoTMixin,) if config.is_groupwise else ()
        return Contribution(
            attrs={"restrict_value_float_to_int_impl": solve_float_to_int_impl_from_enum(rounding)},
            bases=bases)


class ZeroPointComponent(Component):
    """Generic zero-point wiring, used by the weight builder. Symmetric quantizers
    get a fixed zero zero-point (and the max-abs scale stats op); asymmetric
    quantizers get the stats-based asymmetric zero-point mixin. Input builders
    substitute :class:`InputZeroPointComponent`, which subclasses this to add the
    activation-specific tuning (zero_point_impl_type, stats ops, runtime/dynamic
    zero-point)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return self.build_asym(config) if config.is_asym else self.build_sym(config)

    def build_sym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "zero_point_impl": ZeroZeroPoint,
                "scaling_stats_op": _sym_scaling_stats_op(config, StatsOp.MAX)})

    def build_asym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(bases=(AsymmetricZeroPointMixin,))


class SolverComponent(Component):
    """Template: assemble the solver / proxy Contribution (the solver / float base
    goes in ``bases``, the proxy class in ``attrs``). Subclasses supply only the
    kind-specific base class (:meth:`_base`) and proxy class (:meth:`_proxy`)."""

    def build(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={"proxy_class": self._proxy(config)}, bases=(self._base(config),))

    @abstractmethod
    def _base(self, config: QuantizerConfig) -> Type:
        ...

    @abstractmethod
    def _proxy(self, config: QuantizerConfig) -> Type:
        ...


class WeightSolverComponent(SolverComponent):
    """Weight solver, float base and proxy class. The scale implementation type is
    provided by the base :class:`ScaleComponent` (default STATS)."""

    def validate(self, config: QuantizerConfig) -> None:
        # Weights are parameters: DYNAMIC (per-forward) and no_scale (None) are
        # activation-only scale modes.
        if config.scaling_impl_type in (ScalingImplType.DYNAMIC, None):
            raise ValueError("Weight quantizers require a static scale (not DYNAMIC / no_scale).")

    def _base(self, config: QuantizerConfig) -> Type:
        return WeightQuantSolver if config.is_int else ScaledFloatWeightBase

    def _proxy(self, config: QuantizerConfig) -> Type:
        return self._int_proxy(config) if config.is_int else self._float_proxy(config)

    def _int_proxy(self, config: QuantizerConfig) -> Type:
        return GroupwiseWeightQuantProxyFromInjector if config.is_groupwise \
            else WeightQuantProxyFromInjector

    def _float_proxy(self, config: QuantizerConfig) -> Type:
        return GroupwiseWeightFloatQuantProxyFromInjector if config.is_groupwise \
            else WeightFloatQuantProxyFromInjector


class IntQuantComponent(Component):
    """Signedness / narrow-range / zero-point enable for integer quantizers (no-op
    for float, whose signedness comes from the float base).

    The symmetric narrow-range policy is kind-specific (weights use a narrow range
    -> NarrowIntQuant; activations do not -> IntQuant) and supplied by subclasses
    via :attr:`sym_narrow_range`.
    """

    sym_narrow_range: bool

    def build(self, config: QuantizerConfig) -> Contribution:
        if not config.is_int:
            return Contribution()
        return self.build_asym(config) if config.is_asym else self.build_sym(config)

    def build_sym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(attrs={"signed": True, "narrow_range": self.sym_narrow_range})

    def build_asym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(
            attrs={
                "signed": False, "narrow_range": False, "quantize_zero_point": True})


class WeightIntQuantComponent(IntQuantComponent):
    """Integer weight tuning: symmetric weights use a narrow range (NarrowIntQuant)."""

    sym_narrow_range = True


# ---------------------------------------------------------------------------
# Input / activation components. These replace the generic scale / zero-point /
# solver / int-quant components in the input builder's component list (rather
# than layering on top), which keeps the number of overridden / dropped keys to
# a minimum.
# ---------------------------------------------------------------------------
class InputIntQuantComponent(IntQuantComponent):
    """Integer activation tuning: activations use a non-narrow range (IntQuant)."""

    sym_narrow_range = False


class InputScaleComponent(ScaleComponent):
    """Activation scale wiring. Reuses :class:`ScaleComponent`'s base (the
    ``scaling_impl_type`` carried by the config, resolved by the act solver) and
    layers the activation-only overrides each mode needs:

      * static  (``PARAMETER_FROM_STATS``): runtime-percentile stats attrs;
      * dynamic (``DYNAMIC``): the per-forward stats view / broadcast reshape;
      * no_scale (``None``): drop the (now-unused) scale attributes (float-only,
        :class:`FloatActBase`).

    Any other solver-supported ``scaling_impl_type`` is passed through unchanged
    (base only); for a *symmetric* such input the caller must supply
    ``scaling_stats_op`` via ``kwargs``. This component owns the *symmetric*
    scale-stats op for the static / dynamic modes (int static uses a one-sided
    percentile, everything else uses max); the asymmetric scale-stats op is owned
    by :class:`InputZeroPointComponent` / the asymmetric mixin.
    """

    def validate(self, config: QuantizerConfig) -> None:
        # Groupwise activations are recomputed per-forward from the tensor's
        # per-group stats; brevitas has no static / no_scale groupwise activation
        # quantizer, so groupwise requires a dynamic scale.
        if config.is_groupwise and not config.is_dynamic:
            raise ValueError("Groupwise activation quantization requires a dynamic scale.")

    def build(self, config: QuantizerConfig) -> Contribution:
        # Base: scaling_impl_type = config.scaling_impl_type (act solver resolves it).
        contribution = super().build(config)
        if config.is_static:
            contribution += self._static_overrides(config)
        elif config.is_dynamic:
            contribution += self._dynamic_overrides(config)
        elif config.is_no_scale:
            contribution += self._no_scale_overrides(config)
        return contribution

    def _static_overrides(self, config: QuantizerConfig) -> Contribution:
        attrs: Dict[str, Any] = {
            "high_percentile_q": 99.999,
            "collect_stats_steps": 300,}
        if config.is_sym:
            # Static int activations use a one-sided percentile scale; static float
            # uses AbsMax (from ScaledFloatActBase). A signed scale upgrades either
            # to SIGNED_MAX (the percentile attrs above are then left unused, as in
            # the reference signed-scale path).
            default = StatsOp.PERCENTILE if config.is_int else StatsOp.MAX
            attrs["scaling_stats_op"] = _sym_scaling_stats_op(config, default)
        return Contribution(attrs=attrs)

    def _dynamic_overrides(self, config: QuantizerConfig) -> Contribution:
        # scaling_impl_type=DYNAMIC (from the base) is resolved by the act solver to
        # RuntimeDynamicGroupStatsScaling (per-group) or RuntimeDynamicStatsScaling
        # (per-tensor / per-row). Per-group reads group_size/group_dim/input_view
        # from the groupwise act solver; per-tensor / per-row supply the stats view
        # and (for per-tensor) the broadcastable reshape below.
        attrs: Dict[str, Any] = {}
        if not config.is_groupwise:
            if config.scaling_granularity == ScalingPerOutputType.TENSOR:
                attrs["scaling_stats_input_view_shape_impl"] = OverTensorView
                attrs["dynamic_scaling_broadcastable_fn"] = lambda x, shape: x.view(SCALAR_SHAPE)
            else:  # per-row (CHANNEL)
                attrs["scaling_stats_input_view_shape_impl"] = OverOutputFeaturesView
        if config.is_sym:
            attrs["scaling_stats_op"] = _sym_scaling_stats_op(config, StatsOp.MAX)
        return Contribution(attrs=attrs)

    def _no_scale_overrides(self, config: QuantizerConfig) -> Contribution:
        # FloatActBase has no scale; drop the scale-related attributes carried by the
        # base (scaling_impl_type=None) and CommonComponent (restrict_scaling_type),
        # mirroring brevitas Fp8e4m3Act.
        return Contribution(drop=("scaling_impl_type", "restrict_scaling_type"))


class InputZeroPointComponent(ZeroPointComponent):
    """Activation zero-point wiring (replaces :class:`ZeroPointComponent` for
    inputs).

    Symmetric activations reuse the base zero zero-point (the sym scale-stats op is
    owned by :class:`InputScaleComponent`, so it is *not* set here). Asymmetric
    activations reuse the base asymmetric mixin and add the static (runtime
    percentile) or dynamic (per-forward) zero-point tuning.
    """

    def build_sym(self, config: QuantizerConfig) -> Contribution:
        return Contribution(attrs={"zero_point_impl": ZeroZeroPoint})

    def build_asym(self, config: QuantizerConfig) -> Contribution:
        base = super().build_asym(config)  # contributes AsymmetricZeroPointMixin
        if config.is_static:
            # Interval percentile scale + runtime-percentile zero-point (mirrors
            # brevitas ShiftedParamFromPercentileUintQuant), overriding the weight
            # AsymmetricZeroPointMixin defaults with the activation ones.
            attrs = {
                "zero_point_impl_type": ZeroPointImplType.PARAMETER_FROM_RUNTIME,
                "zero_point_stats_op": StatsOp.NEG_PERCENTILE_OR_ZERO,
                "low_percentile_q": 0.001,
                "scaling_stats_op": StatsOp.PERCENTILE_INTERVAL,}
        elif config.is_dynamic:
            # Runtime-dynamic zero-point recomputed per-forward; scale (MIN_MAX)
            # comes from the AsymmetricZeroPointMixin.
            attrs = {
                "zero_point_impl":
                    RuntimeDynamicGroupZeroPoint
                    if config.is_groupwise else RuntimeDynamicStatsZeroPoint}
        else:
            attrs = {}
        # Fold the activation tuning on top of the base asymmetric mixin.
        return base + Contribution(attrs=attrs)


class InputSolverComponent(SolverComponent):
    """Activation solver, float base and proxy class (replaces
    :class:`WeightSolverComponent`). Contributed last so the solver / float base
    sits at the bottom of the MRO, matching the reference activation quantizers."""

    def _base(self, config: QuantizerConfig) -> Type:
        if config.is_int:
            return ActQuantSolver
        # NO_SCALE uses FloatActBase (no scale), otherwise ScaledFloatActBase
        # (which brings the act solver and a stats scale).
        return FloatActBase if config.is_no_scale else ScaledFloatActBase

    def _proxy(self, config: QuantizerConfig) -> Type:
        return self._int_proxy(config) if config.is_int else self._float_proxy(config)

    def _int_proxy(self, config: QuantizerConfig) -> Type:
        # Static / no_scale use the plain proxy; groupwise requires dynamic, which is
        # enforced by InputScaleComponent.validate, so (False, True) never reaches here.
        match (config.is_dynamic, config.is_groupwise):
            case (False, _):
                return ActQuantProxyFromInjector
            case (True, True):
                return GroupwiseActQuantProxyFromInjector
            case (True, False):
                return DynamicActQuantProxyFromInjector

    def _float_proxy(self, config: QuantizerConfig) -> Type:
        # Static / no_scale float use the plain float act proxy.
        match (config.is_dynamic, config.is_groupwise):
            case (False, _):
                return ActFloatQuantProxyFromInjector
            case (True, True):
                return GroupwiseActFloatQuantProxyFromInjector
            case (True, False):
                return DynamicActFloatQuantProxyFromInjector
