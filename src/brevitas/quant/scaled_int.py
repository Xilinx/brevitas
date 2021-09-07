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

from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver


__all__ = [
    'IntBias',
    'Int8Bias',
    'Int16Bias',
    'Int24Bias',
    'Int32Bias',
    'Int8BiasPerTensorFloatInternalScaling',
    'Int8ActPerTensorFloatMinMaxInit',
    'Uint8ActPerTensorFloatMaxInit',
    'Int8ActPerTensorFloat',
    'Int8WeightPerTensorFloat',
    'Uint8ActPerTensorFloat',
    'TruncTo8bit',
    'Int4WeightPerTensorFloatDecoupled'
]


class Int8ActPerTensorFloatMinMaxInit(
    IntQuant, ParamMinMaxInitScaling, PerTensorFloatScaling8bit, ActQuantSolver):
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


class Uint8ActPerTensorFloatMaxInit(
    UintQuant, ParamMinMaxInitScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations quantizer with learned floating-point scale factor
    initialized from a user-defined max val.

    Examples:
        >>> from brevitas.nn import QuantHardTanh
        >>> act = QuantHardTanh(act_quant=Uint8ActPerTensorFloatMaxInit, max_val=.5)
        >>> act.quant_act_scale() * 255
        tensor(0.5000, grad_fn=<MulBackward0>)
    """
    min_val = 0.0


class IntBias(IntQuant, BiasQuantSolver):
    """
    Signed int bias quantizer with bit-width and scale factor equal to the bit-width and the scale
    factor of the accumulator the bias is added to.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=IntBias)
    """
    tensor_clamp_impl = TensorClamp
    requires_input_scale = True
    requires_input_bit_width = True


class Int8Bias(IntBias):
    """
    8-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8Bias)
    """
    bit_width = 8
    requires_input_bit_width = False


class Int16Bias(IntBias):
    """
    16-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 16
    requires_input_bit_width = False


class Int24Bias(IntBias):
    """
    24-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 24
    requires_input_bit_width = False


class Int32Bias(IntBias):
    """
    32-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 32
    requires_input_bit_width = False


class Int8BiasPerTensorFloatInternalScaling(
    IntQuant, MaxStatsScaling, PerTensorFloatScaling8bit, BiasQuantSolver):
    """
    8-bit per-tensor signed int bias quantizer with floating-point scale factor computed from
    backpropagated statistics of the bias tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFloatInternalScaling)
    """
    requires_input_scale = False
    requires_input_bit_width = False


class Int8WeightPerTensorFloat(
    NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit, WeightQuantSolver):
    """
    8-bit narrow per-tensor signed int weight quantizer with floating-point scale factor computed
    from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFloat)
    """
    pass


class Int8ActPerTensorFloat(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    """
    8-bit per-tensor signed int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloat)
    """
    pass


class Uint8ActPerTensorFloat(
    UintQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling8bit, ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloat)
    """
    pass


class TruncTo8bit(IntTrunc, TruncQuantSolver):
    """
    8-bit signed int truncator that preserves the input scale factor and zero-point.

    Examples:
        >>> from brevitas.nn import QuantAvgPool2d
        >>> pool = QuantAvgPool2d(kernel_size=(3, 3), trunc_quant=TruncTo8bit)
    """
    bit_width = 8


class Int4WeightPerTensorFloatDecoupled(WeightPerTensorFloatDecoupledL2Param):
    """
    Experimental narrow per-tensor signed int weight quantizer with decoupled L2,inf
    normalization and learned scaling. Especially suited for the challenging scenario of
    per-tensor low bit-width quantization of depthwise separable weights when retraining from a
    pretrained floating-point model.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> m = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int4WeightPerTensorFloatDecoupled)
    """
    bit_width = 4