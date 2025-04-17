# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import *
from brevitas.quant.base import HQOSymmetricScale
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver

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
    'Int8WeightPerChannelFloat',
    'Uint8ActPerTensorFloat',
    'Int8ActPerTensorFloatMSE',
    'Uint8ActPerTensorFloatMSE',
    'Int8WeightPerTensorFloatMSE',
    'Int8WeightPerChannelFloatMSE',
    'TruncTo8bit',
    'RoundTo8bit',
    'ShiftRoundSaturateTo8bit',
    'Int4WeightPerTensorFloatDecoupled',
    'Int8WeightPerChannelFloatDecoupled',
    'Uint8ActPerTensorFloatBatchQuant1d',
    'Int8ActPerTensorFloatBatchQuant1d',
    'Uint8ActPerTensorFloatBatchQuant2d',
    'Int8ActPerTensorFloatBatchQuant2d',
    'Int8AccumulatorAwareWeightQuant',
    'Int8AccumulatorAwareZeroCenterWeightQuant',
    'Int8WeightNormL2PerChannelFixedPoint']


class Int8ActPerTensorFloatMinMaxInit(IntQuant,
                                      ParamMinMaxInitScaling,
                                      PerTensorFloatScaling8bit,
                                      ActQuantSolver):
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


class Uint8ActPerTensorFloatMaxInit(UintQuant,
                                    ParamMinMaxInitScaling,
                                    PerTensorFloatScaling8bit,
                                    ActQuantSolver):
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


class Int8Bias(IntBias):
    """
    8-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8Bias)
    """
    bit_width = 8


class Int16Bias(IntBias):
    """
    16-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 16


class Int24Bias(IntBias):
    """
    24-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 24


class Int32Bias(IntBias):
    """
    32-bit signed int bias quantizer with scale factor equal to the scale factor of the accumulator
    the bias is added to, so typically quant_input_scale * quant_weight_scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int16Bias)
    """
    bit_width = 32


class Int8BiasPerTensorFloatInternalScaling(IntQuant,
                                            MaxStatsScaling,
                                            PerTensorFloatScaling8bit,
                                            BiasQuantSolver):
    """
    8-bit per-tensor signed int bias quantizer with floating-point scale factor computed from
    backpropagated statistics of the bias tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFloatInternalScaling)
    """
    requires_input_scale = False


class Int8WeightPerTensorFloat(NarrowIntQuant,
                               MaxStatsScaling,
                               PerTensorFloatScaling8bit,
                               WeightQuantSolver):
    """
    8-bit narrow per-tensor signed int weight quantizer with per-tensor floating-point scale factor computed
    from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFloat)
    """
    pass


class Int8WeightPerTensorFloatMSE(MSESymmetricScale, Int8WeightPerTensorFloat):
    """
    8-bit narrow per-tensor signed int weight quantizer with a learned per-tensor floating-point scale factor
    initialized from MSE local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFloatMSE)
    """
    pass


class Int8WeightPerChannelFloat(NarrowIntQuant,
                                MaxStatsScaling,
                                PerChannelFloatScaling8bit,
                                WeightQuantSolver):
    """
    8-bit narrow per-tensor signed int weight quantizer with per-channel floating-point scale factor computed
    from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerChannelFloat)
    """
    pass


class Int8WeightPerChannelFloatMSE(MSESymmetricScale, Int8WeightPerChannelFloat):
    """
    8-bit narrow per-tensor signed int weight quantizer with a learned per-channel floating-point scale factor
    initialized from MSE local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerChannelFloatMSE)
    """
    pass


class Int8ActPerTensorFloat(IntQuant,
                            ParamFromRuntimePercentileScaling,
                            PerTensorFloatScaling8bit,
                            ActQuantSolver):
    """
    8-bit per-tensor signed int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloat)
    """
    pass


class Int8ActPerTensorFloatMSE(MSESymmetricScale, Int8ActPerTensorFloat):
    """
    8-bit per-tensor signed int activations quantizer with learned floating-point scale factor
    initialized from MSE local loss.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloatMSE)
    """
    pass


class Uint8ActPerTensorFloat(UintQuant,
                             ParamFromRuntimePercentileScaling,
                             PerTensorFloatScaling8bit,
                             ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations quantizer with learned floating-point scale factor
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloat)
    """
    pass


class Uint8ActPerTensorFloatMSE(MSESymmetricScale, Uint8ActPerTensorFloat):
    """
    8-bit per-tensor unsigned int activations quantizer with learned floating-point scale factor
    initialized from MSE local loss.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloatMSE)
    """
    pass


class TruncTo8bit(TruncQuantSolver):
    """
    8-bit int truncator that preserves most-significant bits and zero-point.

    Examples:
        >>> from brevitas.nn import TruncAvgPool2d
        >>> pool = TruncAvgPool2d(kernel_size=(3, 3), trunc_quant=TruncTo8bit)
    """
    bit_width = 8
    quant_type = 'int'
    bit_width_impl_type = 'const'
    float_to_int_impl_type = 'floor'
    trunc_scaling_impl_type = 'msb'


class RoundTo8bit(TruncQuantSolver):
    """
    8-bit int truncator with rounding that preserves most-significant bits and zero-point.

    Examples:
        >>> from brevitas.nn import TruncAvgPool2d
        >>> pool = TruncAvgPool2d(kernel_size=(3, 3), trunc_quant=RoundTo8bit)
    """
    bit_width = 8
    quant_type = 'int'
    bit_width_impl_type = 'const'
    float_to_int_impl_type = 'round'
    trunc_scaling_impl_type = 'msb'


class ShiftRoundSaturateTo8bit(TruncQuantSolver,
                               ParamFromRuntimePercentileScaling,
                               PerTensorPoTScaling8bit):
    """
    8-bit shift-round-saturate quantizer which uses statistics to calculate the amount of truncation
    the lest-significant bits and most-significant bits. Zero-point is preserved.

    Examples:
        >>> from brevitas.nn import TruncAvgPool2d
        >>> pool = TruncAvgPool2d(kernel_size=(3, 3), trunc_quant=ShiftRoundSaturateTo8bit)
    """
    bit_width = 8
    quant_type = 'int'
    bit_width_impl_type = 'const'
    float_to_int_impl_type = 'round'
    trunc_scaling_impl_type = 'wrapper'


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


class Int8WeightPerChannelFloatDecoupled(WeightPerChannelFloatDecoupled):
    """
    Experimental narrow per-channel signed int weight quantizer with decoupled Linf
    normalization and learned scaling.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> m = QuantConv2d(4, 4, 3, weight_quant=Int8WeightPerChannelFloatDecoupled)
    """
    bit_width = 8


class Uint8ActPerTensorFloatBatchQuant2d(UintQuant,
                                         BatchQuantStatsScaling2d,
                                         PerTensorFloatScaling8bit,
                                         ActQuantSolver):
    """
    8-bit symmetric per-tensor signed int activations quantizer with affine scale based on s * mean(max_i(abs(x))), s learned,
    with i the output channel dimension of a 4-d tensor with shape (N, C, H, W).
    Statistics are accumulated with a true average rather than a moving exponential one.
    Statistics should be recomputed over calibration data at the end of training, as done in:

    https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html
    Bai, Haoping, et al. "Batchquant: Quantized-for-all architecture search with robust quantizer."
    Advances in Neural Information Processing Systems 34 (2021): 1074-1085.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloatBatchQuant2d)
    """
    pass


class Int8ActPerTensorFloatBatchQuant2d(IntQuant,
                                        BatchQuantStatsScaling2d,
                                        PerTensorFloatScaling8bit,
                                        ActQuantSolver):
    """
    8-bit symmetric per-tensor unsigned int activations quantizer with affine scale based on s * mean(max_i(abs(x))), s learned,
    with i the output channel dimension of a 4-d tensor with shape (N, C, H, W).
    Statistics are accumulated with a true average rather than a moving exponential one.
    Statistics should be recomputed over calibration data at the end of training, as done in:

    https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html
    Bai, Haoping, et al. "Batchquant: Quantized-for-all architecture search with robust quantizer."
    Advances in Neural Information Processing Systems 34 (2021): 1074-1085.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloatBatchQuant2d)
    """
    pass


class Uint8ActPerTensorFloatBatchQuant1d(UintQuant,
                                         BatchQuantStatsScaling1d,
                                         PerTensorFloatScaling8bit,
                                         ActQuantSolver):
    """
    8-bit symmetric per-tensor signed int activations quantizer with affine scale based on s * mean(max_i(abs(x))), s learned,
    with i the output channel dimension of a 3-d tensor with shape (N, C, Feat).
    Statistics are accumulated with a true average rather than a moving exponential one.
    Statistics should be recomputed over calibration data at the end of training, as done in:

    https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html
    Bai, Haoping, et al. "Batchquant: Quantized-for-all architecture search with robust quantizer."
    Advances in Neural Information Processing Systems 34 (2021): 1074-1085.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFloatBatchQuant1d)
    """
    pass


class Int8ActPerTensorFloatBatchQuant1d(IntQuant,
                                        BatchQuantStatsScaling1d,
                                        PerTensorFloatScaling8bit,
                                        ActQuantSolver):
    """
    8-bit symmetric per-tensor unsigned int activations quantizer with affine scale based on s * mean(max_i(abs(x))), s learned,
    with i the output channel dimension of a 3-d tensor with shape (N, C, Feat).
    Statistics are accumulated with a true average rather than a moving exponential one.
    Statistics should be recomputed over calibration data at the end of training, as done in:

    https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html
    Bai, Haoping, et al. "Batchquant: Quantized-for-all architecture search with robust quantizer."
    Advances in Neural Information Processing Systems 34 (2021): 1074-1085.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFloatBatchQuant1d)
    """
    pass


class Int8WeightNormL2PerChannelFixedPoint(WeightNormPerChannelFloatDecoupled):
    """
    Experimental 8-bit narrow signed integer quantizer with learned per-channel scaling factors
    and L2 weight normalization based on `A2Q: Accumulator-Aware Quantization with Guaranteed Overflow
    Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig (https://arxiv.org/abs/2308.13504).
    The quantizer learns scaling factors and norm parameter g in the log-float domain with the half-way
    rounding function.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int8WeightNormL2PerChannelFixedPoint)
        >>> conv.quant_weight()
    """
    bit_width = 8


class Int8AccumulatorAwareWeightQuant(AccumulatorAwareWeightQuant):
    """
    Experimental 8-bit narrow signed accumulator-aware integer quantizer with learned per-channel
    scaling factors based on `A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance`
    by I.Colbert, A.Pappalardo, and J.Petri-Koenig (https://arxiv.org/abs/2308.13504). The quantizer
    learns scaling factors s and norm parameter g in the log-float domain with the round-to-zero
    rounding function. The norm is clamped according the specified accumulator bit-width.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int8AccumulatorAwareWeightQuant)
        >>> conv.quant_weight()
    """
    bit_width = 8


class Int8AccumulatorAwareZeroCenterWeightQuant(AccumulatorAwareZeroCenterWeightQuant):
    """
    Experimental 8-bit narrow signed zero-centered accumulator-aware integer weight quantizer with
    learned per-channel scaling factors based on `A2Q+: Improving Accumulator-Aware Weight Quantization`
    by I. Colbert, A. Pappalardo, J. Petri-Koenig, and Y. Umuroglu (https://arxiv.org/abs/2401.10432).
    The quantizer learns scaling factors in the float domain and learns norm parameter g in the log domain
    with the round-to-zero rounding function. The norm is clamped according the specified accumulator
    bit-width using zero-centered weights. The zero-centering is done before rounding and clipping.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int8AccumulatorAwareZeroCenterWeightQuant)
        >>> conv.quant_weight()
    """
    bit_width = 8


class Int8WeightPerTensorFloatHQO(HQOSymmetricScale, Int8WeightPerTensorFloat):
    """
    8-bit narrow per-tensor signed int weight quantizer with per-tensor floating-point scale factor computed
    from HQO local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFloatHQO)
    """
    pass


class Int8WeightPerChannelFloatHQO(HQOSymmetricScale, Int8WeightPerChannelFloat):
    """
    8-bit narrow per-tensor signed int weight quantizer with per-tensor floating-point scale factor computed
    from HQO local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerChannelFloatHQO)
    """
    pass
