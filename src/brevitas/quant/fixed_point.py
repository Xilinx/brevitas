# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.quant.base import *
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver

__all__ = [
    'Int8WeightPerTensorFixedPoint',
    'Int8ActPerTensorFixedPoint',
    'Uint8ActPerTensorFixedPoint',
    'Int8BiasPerTensorFixedPointInternalScaling',
    'Uint8ActPerTensorFixedPointMaxInit']


class Int8WeightPerTensorFixedPoint(NarrowIntQuant,
                                    MaxStatsScaling,
                                    PerTensorPoTScaling8bit,
                                    WeightQuantSolver):
    """
    8-bit narrow per-tensor signed fixed-point weight quantizer with the radix point
    computed from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFixedPoint)
        >>> fc.quant_weight()
    """
    pass


class Int8ActPerTensorFixedPoint(IntQuant,
                                 ParamFromRuntimePercentileScaling,
                                 PerTensorPoTScaling8bit,
                                 ActQuantSolver):
    """
    8-bit per-tensor signed int activations fixed-point quantizer with learned radix point
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
    """
    pass


class Int8ActPerTensorFixedPointMinMaxInit(IntQuant,
                                           ParamMinMaxInitScaling,
                                           PerTensorPoTScaling8bit,
                                           ActQuantSolver):
    """
    8-bit per-tensor signed int activations fixed-point quantizer with learned radix point
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
    """
    pass


class Uint8ActPerTensorFixedPoint(UintQuant,
                                  ParamFromRuntimePercentileScaling,
                                  PerTensorPoTScaling8bit,
                                  ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations fixed-point quantizer with learned radix point
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFixedPoint)
    """
    pass


class Uint8ActPerTensorFixedPointMaxInit(UintQuant,
                                         ParamMinMaxInitScaling,
                                         PerTensorPoTScaling8bit,
                                         ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations quantizer with learned power-of-two scale factor
    initialized from a user-defined max val.

    Examples:
        >>> from brevitas.nn import QuantHardTanh
        >>> act = QuantHardTanh(act_quant=Uint8ActPerTensorFixedPointMaxInit, max_val=.5)
        >>> act.quant_act_scale() * 255
        tensor(0.4980, grad_fn=<MulBackward0>)
    """
    min_val = 0.0


class Int8BiasPerTensorFixedPointInternalScaling(IntQuant,
                                                 MaxStatsScaling,
                                                 PerTensorPoTScaling8bit,
                                                 BiasQuantSolver):
    """
    8-bit per-tensor signed fixed-point bias quantizer with the radix point computed
    from backpropagated statistics of the bias tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFixedPointInternalScaling)
    """
    requires_input_scale = False
    requires_input_bit_width = False


class Int4WeightPerTensorFixedPointDecoupled(WeightPerTensorFloatDecoupledL2Param):
    """
    Experimental 4-bit narrow per-tensor signed fixed-point weight quantizer with quantized L2,inf
    normalization and learned radix point. Suitable for retraining from floating-point
    depthwise separable weights.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int4WeightPerTensorFixedPointDecoupled)
        >>> conv.quant_weight()
    """
    bit_width = 4
    restrict_scaling_impl = PowerOfTwoRestrictValue
    int_scaling_impl = PowerOfTwoIntScaling
    restrict_value_float_to_int_impl = CeilSte


class Int8WeightNormL2PerChannelFixedPoint(WeightNormPerChannelFloatDecoupled):
    """
    Experimental 8-bit narrow signed integer quantizer with learned per-channel scaling factors
    and L2 weight normalization based on `Quantized Neural Networks for Low-Precision Accumulation
    with Guaranteed Overflow Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig
    (https://arxiv.org/abs/2301.13376). The quantizer learns scaling factors in the float domain and
    learns vector parameter g in the log domain with the half-way rounding function. Suitable for
    retraining from floating-point depthwise separable weights.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int8WeightNormL2PerChannelFixedPoint)
        >>> conv.quant_weight()
    """
    bit_width = 8


class Int8AccumulatorAwareWeightQuant(AccumulatorAwareWeightQuant):
    """
    Experimental 8-bit narrow signed accumulator-aware integer quantizer with learned per-channel
    scaling factors based on `Quantized Neural Networks for Low-Precision Accumulation with Guaranteed
    Overflow Avoidance` by I.Colbert, A.Pappalardo, and J.Petri-Koenig (https://arxiv.org/abs/2301.13376).
    The quantizer learns scaling factors in the float domain and learns vector parameter g in the log
    domain with the round-to-zero rounding function. The norm is clamped according the the specified
    accumulator bit-width. Suitable for retraining from floating-point depthwise separable weights.

    Examples:
        >>> from brevitas.nn import QuantConv2d
        >>> conv = QuantConv2d(4, 4, 3, groups=4, weight_quant=Int8AccumulatorAwareWeightQuant)
        >>> conv.quant_weight()
    """
    bit_width = 8
