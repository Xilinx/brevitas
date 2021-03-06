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


__all__ = [
    'Int8Bias',
    'Int8BiasPerTensorFloatInternalScaling',
    'Int8ActPerTensorFloatMinMaxInit',
    'Int8ActPerTensorFloat',
    'Int8WeightPerTensorFloat',
    'Uint8ActPerTensorFloat',
    'TruncTo8bit'
]


class Int8ActPerTensorFloatMinMaxInit(
    IntQuant, ParamMinMaxInitScaling, PerTensorFloatScaling8bit):
    """
    8-bit per-tensor signed int activations quantizer with learned floating-point scale factor
    initialized from user-defined min and max values.

    Examples:
        >>> from brevitas.nn import QuantHardTanh
        >>> act = QuantHardTanh(act_quant=Int8ActPerTensorFloatMinMaxInit, min_val=-.5, max_val=.5)
        >>> act.quant_act_scale() * - 128
        tensor(-0.5000, grad_fn=<MulBackward0>)
        >>> act.quant_act_scale() * 127
        tensor(0.4961, grad_fn=<MulBackward0>)
    """
    pass


class Int8Bias(IntQuant):
    """
    8-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8Bias)
    """
    bit_width = 8


class Int8BiasPerTensorFloatInternalScaling(
    IntQuant, MaxStatsScaling, PerTensorFloatScaling8bit):
    """
    8-bit per-tensor signed int bias quantizer with floating-point scale factor computed from
    backpropagated statistics of the bias tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFloatInternalScaling)
    """
    pass


class Int8WeightPerTensorFloat(
    NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit):
    """
    8-bit narrow per-tensor signed int weight quantizer with floating-point scale factor computed
    from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, weight_quant=Int8WeightPerTensorFloat)
    """
    pass


class Int8ActPerTensorFloat(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit):
    """
    8-bit per-tensor signed int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloat)
    """
    pass


class Uint8ActPerTensorFloat(
    UintQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit):
    """
    8-bit per-tensor unsigned int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloat)
    """
    pass


class TruncTo8bit(IntTrunc):
    """
    8-bit signed int truncator that preserves the input scale factor and zero-point.

    Examples:
        >>> from brevitas.nn import QuantAvgPool2d
        >>> pool = QuantAvgPool2d(kernel_size=(3, 3), trunc_quant=TruncTo8bit)
    """
    bit_width = 8