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


from brevitas.core.quant import *
from brevitas.core.quant import QuantType
from brevitas.quant.solver.common import *
from brevitas.quant.solver.parameter import *
from brevitas.inject import ExtendedInjector, value, this


__all__ = [
    'SolveWeightTensorQuantFromEnum',
    'SolveWeightScalingStatsInputConcatDimFromModule',
    'SolveWeightScalingPerOutputChannelShapeFromModule',
    'WeightQuantSolver'
]

class SolveWeightTensorQuantFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_quant(quant_type):
        if quant_type == QuantType.FP:
            return None
        elif quant_type == QuantType.INT:
            return RescalingIntQuant
        elif quant_type == QuantType.TERNARY:
            return TernaryQuant
        elif quant_type == QuantType.BINARY:
            return BinaryQuant
        else:
            raise RuntimeError(f'{quant_type} not recognized.')


class SolveWeightScalingStatsInputConcatDimFromModule(ExtendedInjector):

    @value
    def scaling_stats_input_concat_dim(module):
        if isinstance(module, tuple):
            assert all(m.output_channel_dim == module[0].output_channel_dim for m in module)
            module = module[0]
        return module.output_channel_dim


class SolveWeightScalingPerOutputChannelShapeFromModule(ExtendedInjector):

    @value
    def scaling_per_output_channel_shape(module):
        if isinstance(module, tuple):
            assert all(m.out_channels == module[0].out_channels for m in module)
            module = module[0]
        shape = [1] * len(module.weight.size())
        shape[module.output_channel_dim] = module.out_channels
        return tuple(shape)


class WeightQuantSolver(
        SolveScalingStatsInputViewShapeImplFromEnum,
        SolveStatsReduceDimFromEnum,
        SolveScalingStatsOpFromEnum,
        SolveBitWidthImplFromEnum,
        SolveTensorQuantFloatToIntImplFromEnum,
        SolveRestrictScalingImplFromEnum,
        SolveIntScalingImplFromEnum,
        SolveParameterScalingImplFromEnum,
        SolveParameterTensorClampImplFromEnum,
        SolveParameterScalingInitFromEnum,
        SolveParameterScalingShape,
        SolveWeightScalingPerOutputChannelShapeFromModule,
        SolveWeightScalingStatsInputConcatDimFromModule,
        SolveWeightTensorQuantFromEnum):
    """
    Translate enum and shape directives to weight-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    pass



