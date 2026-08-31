"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Input/activation quantizer builder (v2): the concrete
:class:`InputQuantizerBuilder` with its ordered component list. Instantiate it via
the shared ``build_quantizer(InputQuantizerBuilder, ...)`` factory (see
:mod:`.builder`).
"""
from typing import List

from brevitas_examples.common.quantizer_builder.builder import QuantizerBuilder
from brevitas_examples.common.quantizer_builder.components import CommonComponent
from brevitas_examples.common.quantizer_builder.components import FormatComponent
from brevitas_examples.common.quantizer_builder.components import InputIntQuantComponent
from brevitas_examples.common.quantizer_builder.components import InputScaleComponent
from brevitas_examples.common.quantizer_builder.components import InputSolverComponent
from brevitas_examples.common.quantizer_builder.components import InputZeroPointComponent
from brevitas_examples.common.quantizer_builder.components import ScaleParamMethodComponent
from brevitas_examples.common.quantizer_builder.components import ScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.components import ZeroPointParamMethodComponent
from brevitas_examples.common.quantizer_builder.core import Component


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
