# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.quant.base import *
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.trunc import TruncQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver

__all__ = [
    'ShiftedUint8ActPerTensorFixedPoint',
    'ShiftedUint8ActPerTensorFloat',
    'ShiftedUint8WeightPerTensorFloat',
    'ShiftedUint8WeightPerChannelFloat']


class ShiftedUint8ActPerTensorFixedPoint(ShiftedParamFromPercentileUintQuant,
                                         ParamFromRuntimePercentileIntervalScaling,
                                         PerTensorPoTScaling8bit,
                                         ActQuantSolver):
    """
    8-bit per-tensor unsigned int fixed-point activations quantizer with
    integer zero point. Both zero-point and scale factors are learned parameters initialized from
    runtime statistics.

        Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=ShiftedUint8ActPerTensorFixedPoint)
    """
    pass


class ShiftedUint8ActPerTensorFloat(ShiftedParamFromPercentileUintQuant,
                                    ParamFromRuntimePercentileIntervalScaling,
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


class ShiftedUint8WeightPerTensorFloat(ShiftedMinUintQuant,
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


class ShiftedUint8WeightPerChannelFloat(ShiftedMinUintQuant,
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
