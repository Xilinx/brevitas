# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.bit_width import MsbClampBitWidth
from brevitas.core.bit_width import RemoveBitwidthParameter
from brevitas.core.quant import PrescaledRestrictIntQuantWithInputBitWidth
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import QuantType
from brevitas.proxy import ClampQuantProxyFromInjector


class SolveClampTensorQuantFromEnum(ExtendedInjector):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return PrescaledRestrictIntQuantWithInputBitWidth
        elif quant_type == QuantType.TERNARY:
            raise RuntimeError(f'{quant_type} not supported for clamping.')
        elif quant_type == QuantType.BINARY:
            raise RuntimeError(f'{quant_type} not supported for clamping.')
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class SolveClampBitWidthImplFromEnum(ExtendedInjector):

    @value
    def bit_width_impl(quant_type):
        if quant_type == QuantType.INT:
            return MsbClampBitWidth
        else:
            return None

    @value
    def bit_width_to_remove_impl(quant_type, bit_width_impl_type):
        if quant_type == QuantType.INT:
            if bit_width_impl_type == BitWidthImplType.CONST:
                return BitWidthConst
            elif bit_width_impl_type == BitWidthImplType.PARAMETER:
                return RemoveBitwidthParameter
            else:
                raise RuntimeError(f'{quant_type} not recognized.')
        else:
            return None


class ClampQuantSolver(SolveClampBitWidthImplFromEnum, SolveClampTensorQuantFromEnum):
    """
    Translate enum directives to clamping-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = ClampQuantProxyFromInjector
