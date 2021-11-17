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
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.quant.solver.common import *
from brevitas.quant.solver.parameter import *
from brevitas.inject import ExtendedInjector, value, this


__all__ = [
    'SolveWeightTensorQuantFromEnum',
    'SolveWeightScalingStatsInputDimsFromModule',
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


class SolveWeightScalingPerOutputChannelShapeFromModule(ExtendedInjector):

    @value
    def out_channels(module):
        if isinstance(module, tuple):
            assert all(m.out_channels == module[0].out_channels for m in module)
            module = module[0]
        return module.out_channels

    @value
    def weight_ndims(module):
        if isinstance(module, tuple):
            assert all(len(m.weight.size()) == len(module[0].weight.size()) for m in module)
            module = module[0]
        return len(module.weight.size())

    @value
    def scaling_per_output_channel_shape(weight_ndims, output_channel_dim, out_channels):
        shape = [1] * weight_ndims
        shape[output_channel_dim] = out_channels
        return tuple(shape)


class SolveWeightScalingStatsInputDimsFromModule(ExtendedInjector):

    #  during per-channel quantization weights are always permuted and reshaped first
    #  such that output channels are dim 0 and the remaining features are dim 1,
    #  along which we concatenate
    @value
    def scaling_stats_input_concat_dim(scaling_per_output_channel):
        if scaling_per_output_channel:
            return 1
        else:
            return 0

    @value
    def permute_dims(module, output_channel_dim):
        if output_channel_dim != 0:
            dims = list(range(0, len(module.weight.shape)))
            dims[0], dims[output_channel_dim] = dims[output_channel_dim], dims[0]
            return tuple(dims)
        else:
            return None

    @value
    def output_channel_dim(module):
        if isinstance(module, tuple):
            assert all(m.output_channel_dim == module[0].output_channel_dim for m in module)
            module = module[0]
        return module.output_channel_dim


class WeightQuantSolver(
        SolveWeightScalingStatsInputDimsFromModule,
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
        SolveWeightTensorQuantFromEnum):
    """
    Translate enum and shape directives to weight-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = WeightQuantProxyFromInjector



