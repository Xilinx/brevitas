"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Foundational abstractions for the quantizer builder: the immutable configuration
(:class:`QuantizerConfig` and its discriminated :data:`FormatConfig`), the
component contract (:class:`Component`) and its output (:class:`Contribution`).

These are the leaf definitions of the builder package: both the concrete
components and the :class:`~.builder.QuantizerBuilder` depend on them, so keeping
them here (rather than in the builder module) makes the dependency
one-directional and avoids circular imports.
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType


@dataclass(frozen=True)
class IntFormatConfig:
    bit_width: int = 8
    # None = use the kind-specific default (weights are narrow, activations are not);
    # set explicitly to override. Only meaningful for symmetric int (asymmetric int
    # is never narrow). Float formats have no narrow-range concept.
    narrow_range: Optional[bool] = None


@dataclass(frozen=True)
class FloatFormatConfig:
    float_quant_format: str  # e.g. "e4m3"; required, no default
    # AutoName enums are unhashable (they define __eq__ without __hash__), which
    # dataclass rejects as a plain default; use default_factory instead.
    float_format: FloatFormat = field(default_factory=lambda: FloatFormat.FLOAT)


FormatConfig = Union[IntFormatConfig, FloatFormatConfig]


@dataclass(frozen=True)
class QuantizerConfig:
    """
    Immutable description of a quantizer along its orthogonal axes.
    """
    # AutoName enums are unhashable, so enum defaults use default_factory (a plain
    # default is rejected by dataclass as "mutable").
    format: FormatConfig
    quant_param_type: QuantParamType = field(default_factory=lambda: QuantParamType.SYM)
    scaling_granularity: ScalingPerOutputType = field(
        default_factory=lambda: ScalingPerOutputType.TENSOR)
    # TODO (pml): Consider adding a check for `scaling_impl_type=None`
    # Optional: None encodes the activation no-scale (float-only) mode.
    scaling_impl_type: Optional[ScalingImplType] = field(
        default_factory=lambda: ScalingImplType.STATS)
    restrict_scaling_type: RestrictValueType = field(default_factory=lambda: RestrictValueType.FP)
    scaling_param_method: ParamMethod = field(default_factory=lambda: ParamMethod.STATS)
    zero_point_param_method: Optional[ParamMethod] = None
    # Caller-supplied namespace overrides (highest precedence, applied over every
    # component's attrs) and base classes (lowest MRO priority, appended after
    # every component's bases).
    extra: Dict[str, Any] = field(default_factory=dict)
    extra_bases: Tuple[Type, ...] = ()

    @property
    def is_int(self) -> bool:
        return isinstance(self.format, IntFormatConfig)

    @property
    def is_float(self) -> bool:
        return isinstance(self.format, FloatFormatConfig)

    @property
    def is_sym(self) -> bool:
        return QuantParamType(self.quant_param_type) == QuantParamType.SYM

    @property
    def is_asym(self) -> bool:
        return QuantParamType(self.quant_param_type) == QuantParamType.ASYM

    @property
    def is_groupwise(self) -> bool:
        return ScalingPerOutputType(self.scaling_granularity) == ScalingPerOutputType.GROUP

    @property
    def is_static(self) -> bool:
        return self.scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS

    @property
    def is_dynamic(self) -> bool:
        return self.scaling_impl_type == ScalingImplType.DYNAMIC

    @property
    def is_no_scale(self) -> bool:
        return self.scaling_impl_type is None

    @property
    def is_power_of_two(self) -> bool:
        return RestrictValueType(self.restrict_scaling_type) == RestrictValueType.POWER_OF_TWO

    @property
    def is_signed_scale(self) -> bool:
        # A signed scale is expressed through the restrict-value axis (SIGNED_FP).
        # It is only meaningful for symmetric quantizers; asymmetric quantizers
        # ignore it (see the symmetric scale-stats-op components).
        return RestrictValueType(self.restrict_scaling_type) == RestrictValueType.SIGNED_FP

    # TODO (pml): Lift these contraints as, at least, a subset of these combinations should
    # be supported in the future (e.g. float + asymmetric quantization)
    def __post_init__(self) -> None:
        if self.is_float and not self.is_sym:
            raise ValueError("Float quantizers only support symmetric quantization.")
        if self.is_float and self.zero_point_param_method is not None:
            raise ValueError("Float quantizers do not support a zero-point param method.")
        if self.is_sym and self.zero_point_param_method is not None:
            raise ValueError(
                "Zero point parameter method is not applicable for symmetric quantization.")
        if self.is_no_scale and not self.is_float:
            raise ValueError("no_scale quantization is only supported for float quant_type.")
        # Groupwise power-of-two scaled float (MX) is only defined for the OCP
        # format (it relies on FpOCPMixin's inf/nan values for the mantissa bias).
        if (self.is_float and self.is_power_of_two and self.is_groupwise and
                self.format.float_format != FloatFormat.OCP):
            raise ValueError(
                "Groupwise power-of-two scaled float quantizers (MX) are only "
                "supported for FloatFormat.OCP.")
        # MSE/HQO SCALE incompatible with dynamic scale
        if self.is_dynamic and self.scaling_param_method != ParamMethod.STATS:
            raise ValueError(
                "Dynamic scale quantization not supported with non-STATS scaling_param_method (MSE/HQO)."
            )
        # MSE/HQO ZERO_POINT incompatible with dynamic scale
        if self.is_asym and self.is_dynamic and self.zero_point_param_method is not None:
            raise ValueError(
                "Dynamic zero-point quantization not supported with non-None zero_point_param_method (MSE/HQO)."
            )
        # HQO is incomptible with non-integer quantization
        if self.scaling_param_method == ParamMethod.HQO and not self.is_int:
            raise ValueError("HQO scaling_param_method is only supported for integer quantization.")
        # An MSE scale and an HQO zero-point mix incompatible input-view shapes
        # across the two local-loss optimizers (no reference quantizer pairs them,
        # and HalfQuadraticOptimizerZeroPoint crashes on the shape mismatch).
        if (self.scaling_param_method == ParamMethod.MSE and
                self.zero_point_param_method == ParamMethod.HQO):
            raise ValueError(
                "MSE scaling_param_method is incompatible with an HQO zero_point_param_method.")
        # For groupwise quantization, `group_dim` and `group_size` must be specified in `extra`
        if self.is_groupwise and ('group_dim' not in self.extra or 'group_size' not in self.extra):
            raise ValueError(
                "For groupwise quantization, `group_dim` and `group_size` must be specified in `extra`."
            )


@dataclass(frozen=True)
class Contribution:
    """A component's delta to the injector: the only type that knows about the
    ``(namespace, base_classes)`` pair. Components return these and the builder
    folds them."""
    attrs: Dict[str, Any] = field(default_factory=dict)
    bases: Tuple[Type, ...] = ()
    # Namespace keys to remove after applying ``attrs`` (rare; e.g. no_scale float
    # drops the scale-related attributes carried by earlier components).
    drop: Tuple[str, ...] = ()

    def __add__(self, other: "Contribution") -> "Contribution":
        """Fold ``other`` on top of ``self``: later ``attrs`` win, ``bases`` are
        appended (so ``self``'s bases sit first in the MRO) and ``drop`` sets are
        unioned. ``drop`` is *not* applied here (the builder applies it last, after
        every contribution is folded and ``config.extra`` is merged), so a
        component can remove an attribute regardless of ordering."""
        return Contribution(
            attrs={
                **self.attrs, **other.attrs},
            bases=self.bases + other.bases,
            drop=self.drop + other.drop)

    @classmethod
    def merge(cls, contributions: Iterable["Contribution"]) -> "Contribution":
        """Fold an ordered iterable of contributions into a single one."""
        result = cls()
        for contribution in contributions:
            result = result + contribution
        return result


class Component(ABC):
    """One axis of the quantizer. Reads what it needs from the config (Context
    Object) and returns a :class:`Contribution`."""

    @abstractmethod
    def build(self, config: QuantizerConfig) -> Contribution:
        ...

    def validate(self, config: QuantizerConfig) -> None:
        """Raise ``ValueError`` on unsupported axis combinations for this component
        (default: no constraints). Run by the builder before assembly."""
        pass


def config_from_flat_args(
        quant_type: Union[str, QuantType],
        *,
        quant_param_type: QuantParamType = QuantParamType.SYM,
        bit_width: int = 8,
        scaling_impl_type: Optional[ScalingImplType] = ScalingImplType.STATS,
        scaling_per_output_type: ScalingPerOutputType = ScalingPerOutputType.TENSOR,
        restrict_scaling_type: RestrictValueType = RestrictValueType.FP,
        scaling_param_method: ParamMethod = ParamMethod.STATS,
        zero_point_param_method: Optional[ParamMethod] = None,
        float_format: Optional[FloatFormat] = None,
        float_quant_format: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None) -> QuantizerConfig:
    """Assemble a :class:`QuantizerConfig` from the legacy flat quantizer arguments.

    Shared by the weight / input factory shims. For inputs the activation scale
    mode is carried by ``scaling_impl_type`` (PARAMETER_FROM_STATS=static,
    DYNAMIC=dynamic, None=no_scale). The ``format`` axis is discriminated on
    ``quant_type`` into an :class:`IntFormatConfig` or :class:`FloatFormatConfig`.

    Only the common axes are exposed as explicit arguments; less-common knobs
    (e.g. ``narrow_range``, ``scaling_min_val``) are passed through ``kwargs`` and
    applied to the injector as-is.
    """
    if QuantType(quant_type) == QuantType.INT:
        fmt: FormatConfig = IntFormatConfig(bit_width=bit_width)
    else:
        fmt = FloatFormatConfig(
            float_quant_format=float_quant_format,
            float_format=float_format if float_format is not None else FloatFormat.FLOAT)
    return QuantizerConfig(
        format=fmt,
        quant_param_type=quant_param_type,
        scaling_granularity=scaling_per_output_type,
        scaling_impl_type=scaling_impl_type,
        restrict_scaling_type=restrict_scaling_type,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        extra=kwargs or {})
