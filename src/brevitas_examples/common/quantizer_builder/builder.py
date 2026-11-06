"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import config_from_args
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType


class QuantizerBuilder(ABC):

    def __init__(
            self,
            config: QuantizerConfig,
            extra_components: Optional[List[Component]] = None) -> None:
        self.config = config
        self.extra_components: List[Component] = extra_components or []

    @abstractmethod
    def base_components(self) -> List[Component]:
        """The ordered list of components to build a specific class of quantizers.

        Order is authoritative: later contributions' ``attrs`` override earlier
        ones, and their ``bases`` are appended after earlier ones (so earlier
        components sit first in the MRO). It encodes the precedence constraints
        (MSE/HQO injectors before the solver / zero-point).
        """
        ...

    def build_quant_injector(self) -> Type:
        """Fold every component's contribution into a ExtendedInjector that describes
        a quantizer.

        The builder's :meth:`base_components` run first, then any caller-supplied
        :attr:`extra_components` (last, lowest MRO priority / final attribute
        writers).
        """
        merged = self._merged_contribution()
        return type("QuantInjector", self._assembled_bases(merged), self._assembled_attrs(merged))

    def _merged_contribution(self) -> Contribution:
        components = self.base_components() + self.extra_components
        for component in components:
            component.validate(self.config)
        return Contribution.merge(component.build(self.config) for component in components)

    def _assembled_bases(self, merged: Contribution) -> Tuple[Type, ...]:
        # ``config.base_overrides`` are appended after every component's bases, so they
        # sit last in the MRO (lowest priority).
        return merged.bases + tuple(self.config.base_overrides)

    def _assembled_attrs(self, merged: Contribution) -> Dict[str, Any]:
        attrs: Dict[str, Any] = dict(merged.attrs)
        attrs.update(self.config.attr_overrides)
        # TODO (pml): Remove drops as they complicate the builder and are not used in practice
        # Drops are applied last so a component can remove an attribute regardless
        # of whether the component that set it ran before or after it.
        for key in merged.drop:
            attrs.pop(key, None)
        return attrs


def create_quantizer_builder(
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
        attr_overrides: Optional[Dict] = None) -> QuantizerBuilder:
    """
    Minimal inteface to instantiate a :class:`QuantizerBuilder` from a subset of quantizer arguments.
    """
    config = config_from_args(
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
        attr_overrides=attr_overrides)
    return builder_cls(config, extra_components=extra_components)
