"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Foundational abstractions for the quantizer builder: the immutable configuration
(:class:`QuantizerConfig` and its discriminated :data:`FormatConfig`), the
component contract (:class:`Component`) and its output (:class:`Contribution`).

These are the leaf definitions of the builder package: both the concrete
components and the :class:`~.quantizer_builder_v2.QuantizerBuilder` depend on
them, so keeping them here (rather than in the builder module) makes the
dependency one-directional and avoids circular imports.
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType


@dataclass(frozen=True)
class IntFormatConfig:
    bit_width: int = 8


@dataclass(frozen=True)
class FloatFormatConfig:
    float_quant_format: str  # e.g. "e4m3"; required, no default
    # AutoName enums are unhashable (they define __eq__ without __hash__), which
    # dataclass rejects as a plain default; use default_factory instead.
    float_format: FloatFormat = field(default_factory=lambda: FloatFormat.FLOAT)


FormatConfig = Union[IntFormatConfig, FloatFormatConfig]


@dataclass(frozen=True)
class QuantizerConfig:
    """Immutable description of a quantizer along its orthogonal axes.

    ``scale_type`` is input/activation-only; weight builders ignore it.
    """
    # AutoName enums are unhashable, so enum defaults use default_factory (a plain
    # default is rejected by dataclass as "mutable").
    format: FormatConfig
    quant_param_type: QuantParamType = field(default_factory=lambda: QuantParamType.SYM)
    scaling_granularity: ScalingPerOutputType = field(
        default_factory=lambda: ScalingPerOutputType.TENSOR)
    scaling_impl_type: ScalingImplType = field(default_factory=lambda: ScalingImplType.STATS)
    restrict_scaling_type: RestrictValueType = field(default_factory=lambda: RestrictValueType.FP)
    scaling_min_val: Optional[float] = None
    scaling_param_method: ParamMethod = field(default_factory=lambda: ParamMethod.STATS)
    zero_point_param_method: Optional[ParamMethod] = None
    scale_type: ScaleType = field(default_factory=lambda: ScaleType.STATIC)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_int(self) -> bool:
        return isinstance(self.format, IntFormatConfig)

    @property
    def is_float(self) -> bool:
        return isinstance(self.format, FloatFormatConfig)

    # TODO (pml): Lift these contraints as, at least, a subset of these combinations should
    # be supported in the future (e.g. float + asymmetric quantization)
    def __post_init__(self) -> None:
        is_sym = QuantParamType(self.quant_param_type) == QuantParamType.SYM
        is_no_scale = ScaleType(self.scale_type) == ScaleType.NO_SCALE
        is_po2 = self.restrict_scaling_type == RestrictValueType.POWER_OF_TWO
        is_groupwise = self.scaling_granularity == ScalingPerOutputType.GROUP
        if self.is_float and not is_sym:
            raise ValueError("Float quantizers only support symmetric quantization.")
        if self.is_float and self.zero_point_param_method is not None:
            raise ValueError("Float quantizers do not support a zero-point param method.")
        if is_sym and self.zero_point_param_method is not None:
            raise ValueError(
                "Zero point parameter method is not applicable for symmetric quantization.")
        if is_no_scale and not self.is_float:
            raise ValueError("no_scale quantization is only supported for float quant_type.")
        # Groupwise power-of-two scaled float (MX) is only defined for the OCP
        # format (it relies on FpOCPMixin's inf/nan values for the mantissa bias).
        if (self.is_float and is_po2 and is_groupwise and
                self.format.float_format != FloatFormat.OCP):
            raise ValueError(
                "Groupwise power-of-two scaled float quantizers (MX) are only "
                "supported for FloatFormat.OCP.")


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


class Component(ABC):
    """One axis of the quantizer. Reads what it needs from the config (Context
    Object) and returns a :class:`Contribution`."""

    @abstractmethod
    def build(self, config: QuantizerConfig) -> Contribution:
        ...
