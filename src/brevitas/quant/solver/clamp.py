# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from brevitas.proxy import ClampQuantProxyFromInjector
from brevitas.core.quant import PrescaledRestrictIntQuantWithInputBitWidth
from brevitas.core.bit_width import MsbClampBitWidth, RemoveBitwidthParameter
from brevitas.core.bit_width import BitWidthImplType, BitWidthConst
from brevitas.inject import ExtendedInjector, value
from brevitas.inject.enum import QuantType


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


class ClampQuantSolver(
        SolveClampBitWidthImplFromEnum,
        SolveClampTensorQuantFromEnum):
    """
    Translate enum directives to clamping-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = ClampQuantProxyFromInjector