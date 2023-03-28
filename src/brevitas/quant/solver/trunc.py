# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.quant import TruncIntQuant
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import QuantType
from brevitas.proxy import TruncQuantProxyFromInjector
from brevitas.quant.solver.common import SolveBitWidthImplFromEnum
from brevitas.quant.solver.common import SolveTensorQuantFloatToIntImplFromEnum


class SolveTruncTensorQuantFromEnum(ExtendedInjector):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return TruncIntQuant
        elif quant_type == QuantType.TERNARY:
            raise RuntimeError(f'{quant_type} not supported for truncation.')
        elif quant_type == QuantType.BINARY:
            raise RuntimeError(f'{quant_type} not supported for truncation.')
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class TruncQuantSolver(SolveBitWidthImplFromEnum,
                       SolveTensorQuantFloatToIntImplFromEnum,
                       SolveTruncTensorQuantFromEnum):
    """
    Translate enum directives to truncation-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = TruncQuantProxyFromInjector
