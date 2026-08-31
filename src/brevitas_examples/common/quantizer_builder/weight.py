"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Weight quantizer builder (v2): the concrete :class:`WeightQuantizerBuilder` with
its ordered component list. Instantiate it via the shared
``build_quantizer(WeightQuantizerBuilder, ...)`` factory (see :mod:`.builder`).
"""
from typing import List

from brevitas_examples.common.quantizer_builder.builder import QuantizerBuilder
from brevitas_examples.common.quantizer_builder.components import CommonComponent
from brevitas_examples.common.quantizer_builder.components import FormatComponent
from brevitas_examples.common.quantizer_builder.components import ScaleComponent
from brevitas_examples.common.quantizer_builder.components import ScaleParamMethodComponent
from brevitas_examples.common.quantizer_builder.components import ScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.components import WeightIntQuantComponent
from brevitas_examples.common.quantizer_builder.components import WeightSolverComponent
from brevitas_examples.common.quantizer_builder.components import ZeroPointComponent
from brevitas_examples.common.quantizer_builder.components import ZeroPointParamMethodComponent
from brevitas_examples.common.quantizer_builder.core import Component


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
