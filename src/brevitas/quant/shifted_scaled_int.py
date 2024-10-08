# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.inject.enum import ScalingPerOutputType
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.base import *
from brevitas.quant.base import HQOActZeroPoint
from brevitas.quant.base import HQOWeightZeroPoint
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver

__all__ = [
    'ShiftedUint8ActPerTensorFixedPoint',
    'ShiftedUint8ActPerTensorFloat',
    'ShiftedUint8WeightPerTensorFloat',
    'ShiftedUint8WeightPerChannelFloat',
    'ShiftedUint8ActPerTensorFixedPointMSE',
    'ShiftedUint8ActPerTensorFloatMSE',
    'ShiftedUint8WeightPerTensorFloatMSE',
    'ShiftedUint8WeightPerChannelFloatMSE',
    'ShiftedUint8ActPerTensorFloatHQO',
    'ShiftedUint8WeightPerChannelFloatHQO',
    'ShiftedUint8WeightPerTensorFloatHQO']


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


class ShiftedUint8ActPerTensorFixedPointMSE(MSEAsymmetricScale,
                                            MSEActZeroPoint,
                                            ShiftedUint8ActPerTensorFixedPoint):
    """
    8-bit per-tensor unsigned int fixed-point activations quantizer with
    integer zero point. Both zero-point and scale factors are learned parameters initialized from
    MSE local loss.

        Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=ShiftedUint8ActPerTensorFixedPointMSE)
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


class ShiftedUint8ActPerTensorFloatMSE(MSEAsymmetricScale,
                                       MSEActZeroPoint,
                                       ShiftedUint8ActPerTensorFloat):
    """
    8-bit per-tensor unsigned int activations quantizer with floating-point scale factor and
    integer zero point. Both zero-point and scale factors are learned parameters initialized from
    MSE local loss.

        Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=ShiftedUint8ActPerTensorFloatMSE)
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


class ShiftedUint8WeightPerTensorFloatMSE(MSEAsymmetricScale,
                                          MSEWeightZeroPoint,
                                          ShiftedUint8WeightPerTensorFloat):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-tensor scale factor and integer
    zero point. Both zero-point and scale factors are learned parameters initialized from MSE local losses.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerTensorFloatMSE)
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


class ShiftedUint8WeightPerChannelFloatMSE(MSEAsymmetricScale,
                                           MSEWeightZeroPoint,
                                           ShiftedUint8WeightPerChannelFloat):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-channel scale factor and integer
    zero point. Both zero-point and scale factors are learned parameters initialized from MSE local losses.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerChannelFloat)
    """
    pass


class ShiftedUint8WeightPerTensorFloatHQO(HQOWeightZeroPoint, ShiftedUint8WeightPerTensorFloat):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-channel scale factor and integer
    zero point. Zero-point is initialized from HQO local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerTensorFloatHQO)
    """
    quantize_zero_point = False


class ShiftedUint8WeightPerChannelFloatHQO(HQOWeightZeroPoint, ShiftedUint8WeightPerChannelFloat):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-channel scale factor and integer
    zero point. Zero-point is initialized from HQO local loss.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerChannelFloatHQO)
    """
    quantize_zero_point = False


class ShiftedUint8WeightPerGroupFloatHQO(ShiftedUint8WeightPerChannelFloatHQO):
    """
    8-bit per-tensor unsigned int weight quantizer with floating-point per-channel scale factor and integer
    zero point.Zero-point is initialized from HQO local loss.
    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=ShiftedUint8WeightPerChannelFloatHQO)
    """
    group_size = 32
    scaling_per_output_type = ScalingPerOutputType.GROUP
    proxy_class = GroupwiseWeightQuantProxyFromInjector


class ShiftedUint8ActPerTensorFloatHQO(HQOActZeroPoint, ShiftedUint8ActPerTensorFloat):
    """
    8-bit per-tensor unsigned int activations quantizer with floating-point scale factor and
    integer zero point. Zero-point is learned parameter initialized from
    HQO local loss.

        Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=ShiftedUint8ActPerTensorFloatHQO)
    """
    quantize_zero_point = False


class ShiftedUint8WeightGroupQuantFloat(ShiftedUint8WeightPerChannelFloat):
    """
    Block / group / vector signed asymmetric weight quantizer with float scales and zero-points.
    """
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
