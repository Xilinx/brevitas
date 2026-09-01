"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

The :class:`QuantizerBuilder` director. Each concrete (kind-specific) builder
provides its full, ordered component list via :meth:`base_components`; callers may
additionally pass ``extra_components`` to expand or override behaviour. The
director folds every component's contribution into a brevitas injector.
"""
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import config_from_flat_args
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType


class QuantizerBuilder(ABC):
    """Director of the Builder pattern.

    Concrete (kind-specific) builders return the complete, ordered list of
    components from :meth:`base_components`; the order is authoritative and owned
    by each builder. Callers can pass ``extra_components`` to append extra
    components (folded last, so they override attributes and append bases) without
    subclassing.
    """

    def __init__(
            self,
            config: QuantizerConfig,
            extra_components: Optional[List[Component]] = None) -> None:
        self.config = config
        self.extra_components: List[Component] = extra_components or []

    @abstractmethod
    def base_components(self) -> List[Component]:
        """The complete, ordered list of components for this quantizer kind.

        Order is authoritative: later contributions' ``attrs`` override earlier
        ones, and their ``bases`` are appended after earlier ones (so earlier
        components sit first in the MRO). It encodes the precedence constraints
        (param-method injectors before the solver / zero-point; kind tuning last)
        and is guarded by the reference module-hierarchy tests.
        """
        ...

    def build_quant_injector(self) -> Type:
        """Fold every component's contribution into a brevitas injector class.

        The builder's :meth:`base_components` run first, then any caller-supplied
        :attr:`extra_components` (last, lowest MRO priority / final attribute
        writers).
        """
        components = self.base_components() + self.extra_components
        for component in components:
            component.validate(self.config)
        merged = Contribution.merge(component.build(self.config) for component in components)
        return self._assemble(merged)

    def _assemble(self, merged: Contribution) -> Type:
        attrs: Dict[str, Any] = dict(merged.attrs)
        attrs.update(self.config.extra)
        # Drops are applied last so a component can remove an attribute regardless
        # of whether the component that set it ran before or after it.
        for key in merged.drop:
            attrs.pop(key, None)
        return type("QuantInjector", merged.bases, attrs)

    def describe_quantizer(self, resolve: bool = True) -> None:
        """Build the quant injector and print its attributes, dependency kinds,
        and (for ``@value`` functions) the args they require and resolve to."""
        from brevitas_examples.common.quantizer_builder.injector_utils import describe_injector
        describe_injector(self.build_quant_injector(), resolve=resolve)


def build_quantizer(
        builder_cls: Type[QuantizerBuilder],
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
        extra_components: Optional[List[Component]] = None,
        kwargs: Optional[dict] = None) -> QuantizerBuilder:
    """Assemble a :class:`QuantizerConfig` from the legacy flat quantizer arguments
    and return an instance of ``builder_cls`` (e.g. :class:`WeightQuantizerBuilder`
    / :class:`InputQuantizerBuilder`). ``extra_components`` are folded after the
    builder's own components (see :class:`QuantizerBuilder`)."""
    config = config_from_flat_args(
        quant_type,
        quant_param_type=quant_param_type,
        bit_width=bit_width,
        scaling_impl_type=scaling_impl_type,
        scaling_per_output_type=scaling_per_output_type,
        restrict_scaling_type=restrict_scaling_type,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        float_format=float_format,
        float_quant_format=float_quant_format,
        kwargs=kwargs)
    return builder_cls(config, extra_components=extra_components)
