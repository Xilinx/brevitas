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


from brevitas.inject import ExtendedInjector, value
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.quant.solver.common import *
from brevitas.quant.solver.parameter import *
from brevitas.inject.enum import QuantType
from brevitas.core.function_wrapper import Identity
from brevitas.core.quant import RescalingIntQuant, PrescaledRestrictIntQuant
from brevitas.core.quant import PrescaledRestrictIntQuantWithInputBitWidth


__all__ = [
    'BiasQuantSolver',
    'SolveBiasTensorQuantFromEnum',
    'SolveBiasScalingStatsInputConcatDimFromModule',
    'SolveBiasBitWidthImplFromEnum',
    'SolveBiasScalingPerOutputChannelShapeFromModule'
]


class SolveBiasScalingStatsInputConcatDimFromModule(ExtendedInjector):
    scaling_stats_input_concat_dim = 0  # bias has only 1 dimension by definition


class SolveBiasScalingPerOutputChannelShapeFromModule(ExtendedInjector):

    @value
    def scaling_per_output_channel_shape(module):
        if isinstance(module, tuple):
            assert all(m.out_channels == module[0].out_channels for m in module)
            module = module[0]
        return (module.out_channels,)


class SolveBiasBitWidthImplFromEnum(ExtendedInjector):

    @value
    def bit_width_impl(bit_width_impl_type, requires_input_bit_width):
        if not requires_input_bit_width:
            return solve_bit_width_impl_from_enum(bit_width_impl_type)
        else:
            return Identity


class SolveBiasTensorQuantFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_quant(quant_type, requires_input_bit_width, requires_input_scale):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            if not requires_input_bit_width and requires_input_scale:
                return PrescaledRestrictIntQuant
            elif not requires_input_bit_width and not requires_input_scale:
                return RescalingIntQuant
            else:  # requires_input_bit_width == True
                return PrescaledRestrictIntQuantWithInputBitWidth
        elif quant_type == QuantType.TERNARY:
            raise RuntimeError(f'{quant_type} not supported.')
        elif quant_type == QuantType.BINARY:
            raise RuntimeError(f'{quant_type} not supported.')
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class BiasQuantSolver(
        SolveScalingStatsInputViewShapeImplFromEnum,
        SolveParameterScalingShape,
        SolveStatsReduceDimFromEnum,
        SolveScalingStatsOpFromEnum,
        SolveTensorQuantFloatToIntImplFromEnum,
        SolveRestrictScalingImplFromEnum,
        SolveIntScalingImplFromEnum,
        SolveParameterScalingImplFromEnum,
        SolveParameterTensorClampImplFromEnum,
        SolveParameterScalingInitFromEnum,
        SolveBiasBitWidthImplFromEnum,
        SolveBiasScalingPerOutputChannelShapeFromModule,
        SolveBiasScalingStatsInputConcatDimFromModule,
        SolveBiasTensorQuantFromEnum):
    """
    Translate enum directives to bias-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = BiasQuantProxyFromInjector
