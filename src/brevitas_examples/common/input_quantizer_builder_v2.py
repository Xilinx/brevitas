"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Input/activation quantizer builder (v2): the concrete
:class:`InputQuantizerBuilder` with its ordered component list, plus the
``build_input_quantizer`` factory shim that keeps the legacy
``(quant_type, **kwargs)`` signature.
"""
from typing import List
from typing import Optional
from typing import Union

from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quant_builder_components import CommonComponent
from brevitas_examples.common.quant_builder_components import FormatComponent
from brevitas_examples.common.quant_builder_components import InputIntQuantComponent
from brevitas_examples.common.quant_builder_components import InputScaleComponent
from brevitas_examples.common.quant_builder_components import InputSolverComponent
from brevitas_examples.common.quant_builder_components import InputZeroPointComponent
from brevitas_examples.common.quant_builder_components import ScaleParamMethodComponent
from brevitas_examples.common.quant_builder_components import ScaleRestrictComponent
from brevitas_examples.common.quant_builder_components import ZeroPointParamMethodComponent
from brevitas_examples.common.quant_builder_core import Component
from brevitas_examples.common.quant_builder_core import FloatFormatConfig
from brevitas_examples.common.quant_builder_core import IntFormatConfig
from brevitas_examples.common.quant_builder_core import QuantizerConfig
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType
from brevitas_examples.common.quantizer_builder_v2 import QuantizerBuilder


class InputQuantizerBuilder(QuantizerBuilder):
    """Builds an input/activation quantizer injector.

    The input-specific scale / zero-point / solver / int-quant components take the
    place of the generic ones in the ordered list (rather than layering on top),
    which keeps the number of overridden / dropped keys to a minimum. The solver is
    contributed second-to-last (lowest-priority base, matching the reference
    activation quantizers' MRO); the int tuning has the final say over the
    signed / narrow-range attributes.
    """

    def base_components(self) -> List[Component]:
        return [
            ZeroPointParamMethodComponent(),
            ScaleParamMethodComponent(),
            FormatComponent(),
            InputScaleComponent(),
            ScaleRestrictComponent(),
            InputZeroPointComponent(),
            CommonComponent(),
            InputSolverComponent(),
            InputIntQuantComponent(),]


def build_input_quantizer(
        quant_type: Union[str, QuantType],
        *,
        quant_param_type: QuantParamType = QuantParamType.SYM,
        scale_type: ScaleType = ScaleType.STATIC,
        bit_width: int = 8,
        scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
        scaling_per_output_type: ScalingPerOutputType = ScalingPerOutputType.TENSOR,
        restrict_scaling_type: RestrictValueType = RestrictValueType.FP,
        scaling_min_val: Optional[float] = None,
        scaling_param_method: ParamMethod = ParamMethod.STATS,
        zero_point_param_method: Optional[ParamMethod] = None,
        float_format: Optional[FloatFormat] = None,
        float_quant_format: Optional[str] = None,
        extra_components: Optional[List[Component]] = None,
        kwargs: Optional[dict] = None) -> InputQuantizerBuilder:
    """Assemble a :class:`QuantizerConfig` from the legacy flat arguments and
    return an :class:`InputQuantizerBuilder`. ``extra_components`` are folded after
    the builder's own components (see :class:`QuantizerBuilder`)."""
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
        scale_type=scale_type,
        extra=kwargs or {})
    return InputQuantizerBuilder(config, extra_components=extra_components)
