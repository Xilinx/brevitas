"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Weight quantizer builder (v2): the concrete :class:`WeightQuantizerBuilder` with
its ordered component list, plus the ``build_weight_quantizer`` factory shim that
keeps the legacy ``(quant_type, **kwargs)`` signature.
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
from brevitas_examples.common.quant_builder_components import ScaleComponent
from brevitas_examples.common.quant_builder_components import ScaleParamMethodComponent
from brevitas_examples.common.quant_builder_components import ScaleRestrictComponent
from brevitas_examples.common.quant_builder_components import WeightIntQuantComponent
from brevitas_examples.common.quant_builder_components import WeightSolverComponent
from brevitas_examples.common.quant_builder_components import ZeroPointComponent
from brevitas_examples.common.quant_builder_components import ZeroPointParamMethodComponent
from brevitas_examples.common.quant_builder_core import Component
from brevitas_examples.common.quant_builder_core import config_from_flat_args
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder_v2 import QuantizerBuilder


class WeightQuantizerBuilder(QuantizerBuilder):
    """Builds a weight quantizer injector.

    The ordered component list ends with the weight-specific solver and int
    tuning: the solver contributes the lowest-priority base (matching the
    reference weight quantizers' MRO) and the int tuning has the final say over
    the signed / narrow-range attributes.
    """

    def base_components(self) -> List[Component]:
        return [
            ZeroPointParamMethodComponent(),
            ScaleParamMethodComponent(),
            FormatComponent(),
            ScaleComponent(),
            ScaleRestrictComponent(),
            ZeroPointComponent(),
            CommonComponent(),
            WeightSolverComponent(),
            WeightIntQuantComponent(),]


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
        extra_components: Optional[List[Component]] = None,
        kwargs: Optional[dict] = None) -> WeightQuantizerBuilder:
    """Assemble a :class:`QuantizerConfig` from the legacy flat arguments and
    return a :class:`WeightQuantizerBuilder`. ``extra_components`` are folded after
    the builder's own components (see :class:`QuantizerBuilder`)."""
    config = config_from_flat_args(
        quant_type,
        quant_param_type=quant_param_type,
        bit_width=bit_width,
        scaling_impl_type=scaling_impl_type,
        scaling_per_output_type=scaling_per_output_type,
        restrict_scaling_type=restrict_scaling_type,
        scaling_min_val=scaling_min_val,
        scaling_param_method=scaling_param_method,
        zero_point_param_method=zero_point_param_method,
        float_format=float_format,
        float_quant_format=float_quant_format,
        kwargs=kwargs)
    return WeightQuantizerBuilder(config, extra_components=extra_components)
