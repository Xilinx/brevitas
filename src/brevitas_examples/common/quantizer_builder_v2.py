"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Progressive rewrite of the quantizer builder using a Builder + Component design.

This module holds the *engine* (config, contribution, component interface and the
builder that folds components into a brevitas injector). Concrete components and
the weight/input factories are added in later steps; for now this is the
skeleton / abstract layer.

Design in a nutshell:
- ``QuantizerConfig`` describes *what* to build, one immutable value with a
  discriminated ``format`` sub-config (int XOR float) so no field is left unused.
- ``Component`` implementations describe *how* one axis contributes to the
  injector, returning a ``Contribution`` (namespace attrs + base mixins).
- ``QuantizerBuilder`` is the director: it folds an ordered list of components,
  in list order, into the final ``type("QuantInjector", bases, attrs)``. The
  component-list order is authoritative for both attribute precedence (later
  wins) and base-class MRO order.
"""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
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


class QuantizerBuilder(ABC):
    """Director of the Builder pattern.

    ``base_components`` are the components shared by every quantizer, in the order
    that satisfies the precedence constraints. Concrete (kind-specific) builders
    add their own components via :meth:`extra_components`; these run *after* the
    base ones (lowest MRO priority, last attribute writers).
    """

    def __init__(self, config: QuantizerConfig) -> None:
        # Deferred import: the components module imports this module.
        from brevitas_examples.common.quant_builder_components import CommonComponent
        from brevitas_examples.common.quant_builder_components import FormatComponent
        from brevitas_examples.common.quant_builder_components import ScaleComponent
        from brevitas_examples.common.quant_builder_components import ScaleParamMethodComponent
        from brevitas_examples.common.quant_builder_components import ScaleRestrictComponent
        from brevitas_examples.common.quant_builder_components import ZeroPointComponent
        from brevitas_examples.common.quant_builder_components import ZeroPointParamMethodComponent
        self.config = config
        # Base components shared by every quantizer. Each is a named attribute; the
        # build order lives in build_quant_injector (a single, explicit place) so it
        # can't be changed by reordering a list.
        self.zero_point_param_method_component = ZeroPointParamMethodComponent()
        self.scale_param_method_component = ScaleParamMethodComponent()
        self.format_component = FormatComponent()
        self.scale_component = ScaleComponent()
        self.scale_restrict_component = ScaleRestrictComponent()
        self.zero_point_component = ZeroPointComponent()
        self.common_component = CommonComponent()

    @abstractmethod
    def extra_components(self) -> List[Component]:
        """Kind-specific components, appended after the base ones. Typically the
        solver, the kind int-tuning and (for inputs) the scale wiring."""
        ...

    def build_quant_injector(self) -> Type:
        """Build the components in a fixed, explicit order and fold their
        contributions into a brevitas injector class.

        Order is authoritative: later contributions' ``attrs`` override earlier
        ones, and their ``bases`` are appended after earlier ones (so earlier
        components sit first in the MRO). The order below encodes the precedence
        constraints (param-method injectors before the solver / zero-point; kind
        tuning last) and is guarded by the reference module-hierarchy tests.
        """
        contributions = [
            self.zero_point_param_method_component.build(self.config),
            self.scale_param_method_component.build(self.config),
            self.format_component.build(self.config),
            self.scale_component.build(self.config),
            self.scale_restrict_component.build(self.config),
            self.zero_point_component.build(self.config),
            self.common_component.build(self.config),]
        contributions += [component.build(self.config) for component in self.extra_components()]
        return self._assemble(contributions)

    def _assemble(self, contributions: List[Contribution]) -> Type:
        attrs: Dict[str, Any] = {}
        bases: Tuple[Type, ...] = ()
        drops: set = set()
        for contribution in contributions:
            attrs.update(contribution.attrs)
            bases = bases + tuple(contribution.bases)
            drops.update(contribution.drop)
        attrs.update(self.config.extra)
        # Drops are applied last so a component can remove an attribute regardless
        # of whether the component that set it ran before or after it.
        for key in drops:
            attrs.pop(key, None)
        return type("QuantInjector", bases, attrs)

    def describe_quantizer(self, resolve: bool = True) -> None:
        """Build the quant injector and print its attributes, dependency kinds,
        and (for ``@value`` functions) the args they require and resolve to."""
        from brevitas_examples.common.injector_utils import describe_injector
        describe_injector(self.build_quant_injector(), resolve=resolve)
