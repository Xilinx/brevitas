"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.function_wrapper.shape import OverTensorView
from brevitas.core.scaling.runtime import RuntimeDynamicGroupStatsScaling
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import RuntimeDynamicGroupZeroPoint
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.quant.float_base import FloatActBase
from brevitas.quant.float_base import ScaledFloatActBase
from brevitas.quant.solver.act import ActQuantSolver
from brevitas_examples.common.generative.quant_blocks import RuntimeDynamicStatsScaling
from brevitas_examples.common.generative.quant_blocks import RuntimeDynamicStatsZeroPoint
from brevitas_examples.common.quantizer_builder import BaseQuantizerBuilder
from brevitas_examples.common.quantizer_builder import EnumType
from brevitas_examples.common.quantizer_builder import FloatQuantizerBuilder
from brevitas_examples.common.quantizer_builder import IntQuantizerBuilder
from brevitas_examples.common.quantizer_builder import ScaleType
from brevitas_examples.common.quantizer_builder import ZeroPointImplType


# ----------------------------------------------------------------------
# Kind axis: input/activation.
# ----------------------------------------------------------------------
class InputQuantizerBuilder(BaseQuantizerBuilder):
    """Kind axis: quantizes *inputs/activations*.

    Adds the activation-specific ``scale_type`` axis. ``ScaleType.STATIC``
    (runtime percentile scaling stored as a parameter), ``ScaleType.DYNAMIC``
    (scale recomputed per-forward) and ``ScaleType.NO_SCALE`` (float-only, no
    scale) are supported.
    """

    def __init__(
            self, *, scale_type: Union[str, ScaleType] = ScaleType.STATIC, **kwargs: Any) -> None:
        self.scale_type: Union[str, ScaleType] = scale_type
        super().__init__(**kwargs)

    def _is_static(self) -> bool:
        return ScaleType(self.scale_type) == ScaleType.STATIC

    def _is_dynamic(self) -> bool:
        return ScaleType(self.scale_type) == ScaleType.DYNAMIC

    def _is_no_scale(self) -> bool:
        return ScaleType(self.scale_type) == ScaleType.NO_SCALE

    def _is_groupwise(self) -> bool:
        return self.scaling_per_output_type == ScalingPerOutputType.GROUP

    def _build_base_namespace(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = super()._build_base_namespace()
        # Static scaling learns a runtime-percentile scale stored as a parameter;
        # dynamic scaling wires a runtime scaling impl recomputed per-forward. The
        # scaling_stats_op (MAX vs MIN_MAX) is provided by the sym/asym mixins.
        # NO_SCALE quantizers have no scale at all (float-only), so no scale attrs
        # are wired.
        if self._is_static():
            self._build_static_scaling(namespace)
        elif self._is_dynamic():
            self._build_dynamic_scaling(namespace)
        return namespace

    def _build_static_scaling(self, namespace: Dict[str, Any]) -> None:
        # Kind-agnostic static-scaling attributes. The scaling_stats_op is
        # kind-specific (int -> PERCENTILE, float -> AbsMax from ScaledFloatActBase)
        # and is set by the concrete leaf builders.
        namespace['scaling_impl_type'] = ScalingImplType.PARAMETER_FROM_STATS
        namespace['high_percentile_q'] = 99.999
        namespace['collect_stats_steps'] = 300

    def _build_dynamic_scaling(self, namespace: Dict[str, Any]) -> None:
        """Wire the granularity-specific dynamic scaling impl (kind-agnostic).

        Mirrors the reference ``*Dynamic*`` activation quantizers in
        ``brevitas_examples.common.generative.quantizers``.
        """
        # scaling_impl_type is only meaningful for the static parameter path.
        namespace.pop('scaling_impl_type', None)
        if self._is_groupwise():
            # Per-group: RuntimeDynamicGroupStatsScaling reads group_size/group_dim
            # and input_view_impl from the (groupwise) act solver. stats_reduce_dim
            # (over the group block dim) is derived by ActQuantSolver from group_dim.
            namespace['scaling_impl'] = RuntimeDynamicGroupStatsScaling
        else:
            namespace['scaling_impl'] = RuntimeDynamicStatsScaling
            if self.scaling_per_output_type == ScalingPerOutputType.TENSOR:
                namespace['scaling_stats_input_view_shape_impl'] = OverTensorView
                namespace['dynamic_scaling_broadcastable_fn'] = \
                    lambda x, shape: x.view(SCALAR_SHAPE)
            else:  # per-row (CHANNEL)
                namespace['scaling_stats_input_view_shape_impl'] = OverOutputFeaturesView

    def _build_asymmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        base_classes = super()._build_asymmmetric_quantizer(namespace, base_classes)
        if self._is_static():
            # Interval percentile scale + runtime-percentile zero-point for
            # asymmetric static activations (mirrors brevitas
            # ShiftedParamFromPercentileUintQuant). Override the weight AsymMixin
            # defaults with the activation ones. The weight
            # zero_point_stats_input_concat_dim (added by AsymMixin) is left
            # unresolved: ParameterFromRuntimeZeroPoint never requests it.
            namespace['zero_point_impl_type'] = ZeroPointImplType.PARAMETER_FROM_RUNTIME
            namespace['zero_point_stats_op'] = StatsOp.NEG_PERCENTILE_OR_ZERO
            namespace['low_percentile_q'] = 0.001
            namespace['scaling_stats_op'] = StatsOp.PERCENTILE_INTERVAL
        elif self._is_dynamic():
            # Runtime-dynamic zero-point recomputed per-forward; scale (MIN_MAX)
            # comes from the AsymMixin (mirrors brevitas
            # ShiftedUint8DynamicActPer{Tensor,Row,Group}Float).
            namespace['zero_point_impl'] = (
                RuntimeDynamicGroupZeroPoint
                if self._is_groupwise() else RuntimeDynamicStatsZeroPoint)
        return base_classes

    def _build_restrict_param_method(
        self,
        namespace: Dict[str, Any],
        base_classes: Tuple[Type, ...],
        restrict_value_float_to_int_impl_type: EnumType[FloatToIntImplType] = FloatToIntImplType
        .CEIL
    ) -> Tuple[Type, ...]:
        # Non-group dynamic power-of-two activations floor the exponent (mirrors
        # brevitas Int8DynamicActPerRowFixedPoint / FP8e4m3OCPDynamicActPerRowFixedPoint),
        # unlike static po2 activations which ceil it. Groupwise (MX) is already
        # handled as FLOOR by the base implementation.
        if (self._is_dynamic() and self.restrict_scaling_type == RestrictValueType.POWER_OF_TWO and
                self.scaling_per_output_type != ScalingPerOutputType.GROUP):
            restrict_value_float_to_int_impl_type = FloatToIntImplType.FLOOR
        return super()._build_restrict_param_method(
            namespace, base_classes, restrict_value_float_to_int_impl_type)


# ----------------------------------------------------------------------
# Concrete builders: one "kind" x one "format".
# ----------------------------------------------------------------------
class InputIntQuantizerBuilder(InputQuantizerBuilder, IntQuantizerBuilder):
    """Integer input/activation quantizer builder (static or dynamic scaling)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # INPUT_QUANT_MAP has no integer no_scale entry (no_scale is float-only).
        if self._is_no_scale():
            raise ValueError("no_scale input quantization is only supported for float quant_type.")

    def _quant_solver(self) -> Type:
        return ActQuantSolver

    def _proxy_class(self) -> Type:
        if self._is_dynamic():
            if self._is_groupwise():
                return GroupwiseActQuantProxyFromInjector
            return DynamicActQuantProxyFromInjector
        # Groupwise static activation int proxy is not supported yet.
        return ActQuantProxyFromInjector

    def _build_symmmetric_quantizer(
            self, namespace: Dict[str, Any], base_classes: Tuple[Type, ...]) -> Tuple[Type, ...]:
        base_classes = super()._build_symmmetric_quantizer(namespace, base_classes)
        # Static int activations use a one-sided percentile scale.
        if self._is_static():
            namespace['scaling_stats_op'] = StatsOp.PERCENTILE
        # Activations use IntQuant (non-narrow), unlike NarrowIntQuant weights.
        namespace['narrow_range'] = False
        return base_classes


class InputFloatQuantizerBuilder(InputQuantizerBuilder, FloatQuantizerBuilder):
    """Float input/activation quantizer builder (static, dynamic or no scale)."""

    def _solver_base_classes(self) -> Tuple[Type, ...]:
        # NO_SCALE uses FloatActBase (no scale), otherwise ScaledFloatActBase.
        base = FloatActBase if self._is_no_scale() else ScaledFloatActBase
        return super()._solver_base_classes() + (base,)

    def _build_base_namespace(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = super()._build_base_namespace()
        # FloatActBase has no scale, so drop the scale-related attributes carried
        # over from the generic base namespace (mirrors brevitas Fp8e4m3Act).
        if self._is_no_scale():
            for attr in ('scaling_impl_type', 'restrict_scaling_type'):
                namespace.pop(attr, None)
        return namespace

    def _proxy_class(self) -> Type:
        if self._is_dynamic():
            if self._is_groupwise():
                return GroupwiseActFloatQuantProxyFromInjector
            if self.scaling_per_output_type == ScalingPerOutputType.CHANNEL:  # per-row
                return DynamicActFloatQuantProxyFromInjector
            # Per-tensor dynamic float reuses the plain float act proxy (mirrors
            # brevitas Fp8e4m3*DynamicActPerTensorFloat).
            return ActFloatQuantProxyFromInjector
        # Static / no_scale float use the plain float act proxy (FloatActBase's
        # default). Groupwise static activation float proxy is not supported yet.
        return ActFloatQuantProxyFromInjector


# Maps a quant_type to the concrete *input* builder responsible for it.
_INPUT_QUANT_TYPE_BUILDER_MAP = {
    QuantType.INT.value: InputIntQuantizerBuilder,
    QuantType.FP.value: InputFloatQuantizerBuilder,}


def build_input_quantizer(
        quant_type: Union[str, QuantType], *args: Any, **kwargs: Any) -> BaseQuantizerBuilder:
    """Factory returning the appropriate *input* quantizer builder for ``quant_type``.

    Dispatches to :class:`InputIntQuantizerBuilder` (``QuantType.INT``) or
    :class:`InputFloatQuantizerBuilder` (``QuantType.FP``). ``quant_type`` is
    only used to select the builder; the remaining arguments are forwarded
    unchanged to the selected builder's constructor.
    """
    builder_cls = _INPUT_QUANT_TYPE_BUILDER_MAP.get(QuantType(quant_type).value)
    if builder_cls is None:
        raise ValueError(f"No input quantizer builder available for quant_type {quant_type!r}.")
    return builder_cls(*args, **kwargs)
