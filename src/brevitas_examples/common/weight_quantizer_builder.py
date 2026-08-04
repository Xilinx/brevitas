"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
from typing import Any
from typing import Tuple
from typing import Type
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant.float_base import ScaledFloatWeightBase
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas_examples.common.quantizer_builder import BaseQuantizerBuilder
from brevitas_examples.common.quantizer_builder import FloatQuantizerBuilder
from brevitas_examples.common.quantizer_builder import IntQuantizerBuilder


# ----------------------------------------------------------------------
# Kind axis: weights.
# ----------------------------------------------------------------------
class WeightQuantizerBuilder(BaseQuantizerBuilder):
    """Kind axis: quantizes *weights*."""


# ----------------------------------------------------------------------
# Concrete builders: one "kind" x one "format".
# ----------------------------------------------------------------------
class WeightIntQuantizerBuilder(WeightQuantizerBuilder, IntQuantizerBuilder):
    """Integer weight quantizer builder."""

    def _quant_solver(self) -> Type:
        return WeightQuantSolver

    def _proxy_class(self) -> Type:
        if self.scaling_per_output_type == ScalingPerOutputType.GROUP:
            return GroupwiseWeightQuantProxyFromInjector
        return WeightQuantProxyFromInjector


class WeightFloatQuantizerBuilder(WeightQuantizerBuilder, FloatQuantizerBuilder):
    """Float weight quantizer builder."""

    def _solver_base_classes(self) -> Tuple[Type, ...]:
        return super()._solver_base_classes() + (ScaledFloatWeightBase,)

    def _proxy_class(self) -> Type:
        if self.scaling_per_output_type == ScalingPerOutputType.GROUP:
            return GroupwiseWeightFloatQuantProxyFromInjector
        return WeightFloatQuantProxyFromInjector


# Maps a quant_type to the concrete *weight* builder responsible for it.
_QUANT_TYPE_BUILDER_MAP = {
    QuantType.INT.value: WeightIntQuantizerBuilder,
    QuantType.FP.value: WeightFloatQuantizerBuilder,}


def build_weight_quantizer(
        quant_type: Union[str, QuantType], *args: Any, **kwargs: Any) -> BaseQuantizerBuilder:
    """Factory returning the appropriate *weight* quantizer builder for ``quant_type``.

    Dispatches to :class:`WeightIntQuantizerBuilder` (``QuantType.INT``) or
    :class:`WeightFloatQuantizerBuilder` (``QuantType.FP``). ``quant_type`` is
    only used to select the builder; the remaining arguments are forwarded
    unchanged to the selected builder's constructor.
    """
    builder_cls = _QUANT_TYPE_BUILDER_MAP.get(QuantType(quant_type).value)
    if builder_cls is None:
        raise ValueError(f"No quantizer builder available for quant_type {quant_type!r}.")
    return builder_cls(*args, **kwargs)
