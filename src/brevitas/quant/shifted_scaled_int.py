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


from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver

__all__ = [
    'ShiftedUint8ActPerTensorFloat',
    'ShiftedUint8WeightPerTensorFloat',
    'ShiftedUint8WeightPerChannelFloat'
]


class ShiftedUint8ActPerTensorFloat(
    ShiftedRuntimeMinToUintQuant,
    ParamFromRuntimeMinMaxScaling,
    PerTensorFloatScaling8bit,
    ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations quantizer with floating-point scale factor and
    integer zero point. Both zero-point and scale factors are learned parameters initialized from
    runtime statistics.

        Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=ShiftedUint8ActPerTensorFloat)
    """
    pass


class ShiftedUint8WeightPerTensorFloat(
    ShiftedMinUintQuant,
    MinMaxStatsScaling,
    PerTensorFloatScaling8bit,
    WeightQuantSolver):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-tensor scale factor and integer
    zero point. Both zero-point and scale factors are based on backpropagated statistics of the
    weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerTensorFloat)
    """
    pass


class ShiftedUint8WeightPerChannelFloat(
    ShiftedMinUintQuant,
    MinMaxStatsScaling,
    PerChannelFloatScaling8bit,
    WeightQuantSolver):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-channel scale factor and integer
    zero point. Both zero-point and scale factors are based on backpropagated statistics of the
    weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerChannelFloat)
    """
    pass
