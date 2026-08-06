"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

The :class:`QuantizerBuilder` director. It wires the shared base components (from
``quant_builder_core`` / ``quant_builder_components``) and, together with the
kind-specific extras provided by subclasses, folds them into a brevitas injector.
"""
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from brevitas_examples.common.quant_builder_components import CommonComponent
from brevitas_examples.common.quant_builder_components import FormatComponent
from brevitas_examples.common.quant_builder_components import ScaleComponent
from brevitas_examples.common.quant_builder_components import ScaleParamMethodComponent
from brevitas_examples.common.quant_builder_components import ScaleRestrictComponent
from brevitas_examples.common.quant_builder_components import ZeroPointComponent
from brevitas_examples.common.quant_builder_components import ZeroPointParamMethodComponent
from brevitas_examples.common.quant_builder_core import Component
from brevitas_examples.common.quant_builder_core import Contribution
from brevitas_examples.common.quant_builder_core import QuantizerConfig


class QuantizerBuilder(ABC):
    """Director of the Builder pattern.

    The base components shared by every quantizer are named attributes (built in
    :meth:`__init__`); the build order lives in :meth:`build_quant_injector` (a
    single, explicit place). Concrete (kind-specific) builders add their own
    components via :meth:`extra_components`; these run *after* the base ones
    (lowest MRO priority, last attribute writers).
    """

    def __init__(self, config: QuantizerConfig) -> None:
        self.config = config
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
