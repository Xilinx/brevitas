"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Weight quantizer builder (v2): the concrete :class:`WeightQuantizerBuilder` with
its fixed component preset, plus the ``build_weight_quantizer`` factory shim that
keeps the legacy ``(quant_type, **kwargs)`` signature.
"""
from typing import List
from typing import Optional
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quant_builder_components import WeightIntQuantComponent
from brevitas_examples.common.quant_builder_components import WeightSolverComponent
from brevitas_examples.common.quant_builder_core import Component
from brevitas_examples.common.quant_builder_core import FloatFormatConfig
from brevitas_examples.common.quant_builder_core import IntFormatConfig
from brevitas_examples.common.quant_builder_core import QuantizerConfig
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder_v2 import QuantizerBuilder


class WeightQuantizerBuilder(QuantizerBuilder):
    """Builds a weight quantizer injector: the shared base components plus the
    weight-specific solver and int tuning, appended at the end (the solver is the
    lowest-priority base, matching the reference weight quantizers' MRO)."""

    def extra_components(self) -> List[Component]:
        return [WeightSolverComponent(), WeightIntQuantComponent()]


def build_weight_quantizer(
        quant_type: Union[str, QuantType],
        *,
        quant_param_type: QuantParamType = QuantParamType.SYM,
        bit_width: int = 8,
        scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
        scaling_per_output_type: ScalingPerOutputType = ScalingPerOutputType.TENSOR,
        restrict_scaling_type: RestrictValueType = RestrictValueType.FP,
        scaling_min_val: Optional[float] = None,
        scaling_param_method: ParamMethod = ParamMethod.STATS,
        zero_point_param_method: Optional[ParamMethod] = None,
        float_format: Optional[FloatFormat] = None,
        float_quant_format: Optional[str] = None,
        kwargs: Optional[dict] = None) -> WeightQuantizerBuilder:
    """Assemble a :class:`QuantizerConfig` from the legacy flat arguments and
    return a :class:`WeightQuantizerBuilder`."""
    if QuantType(quant_type) == QuantType.INT:
        fmt = IntFormatConfig(bit_width=bit_width)
    else:
        fmt = FloatFormatConfig(
            float_quant_format=float_quant_format,
            float_format=float_format if float_format is not None else FloatFormat.FLOAT)
    config = QuantizerConfig(
        format=fmt,
        quant_param_type=quant_param_type,
        scaling_granularity=scaling_per_output_type,
        scaling_impl_type=scaling_impl_type,
        restrict_scaling_type=restrict_scaling_type,
        scaling_min_val=scaling_min_val,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        extra=kwargs or {})
    return WeightQuantizerBuilder(config)
