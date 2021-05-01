from brevitas.quant.base import *
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver


__all__ = [
    'Int8WeightPerTensorFixedPoint',
    'Int8ActPerTensorFixedPoint',
    'Uint8ActPerTensorFixedPoint',
    'Int8BiasPerTensorFixedPointInternalScaling'
]


class Int8WeightPerTensorFixedPoint(
    NarrowIntQuant, MaxStatsScaling, PerTensorPoTScaling8bit, WeightQuantSolver):
    """
    8-bit narrow per-tensor signed fixed-point weight quantizer with the radix point
    computed from backpropagated statistics of the weight tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=Int8WeightPerTensorFixedPoint)
        >>> fc.quant_weight()
    """
    pass


class Int8ActPerTensorFixedPoint(
    IntQuant, ParamFromRuntimePercentileScaling, PerTensorPoTScaling8bit, ActQuantSolver):
    """
    8-bit per-tensor signed int activations fixed-point quantizer with learned radix point
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)
    """
    pass


class Uint8ActPerTensorFixedPoint(
    UintQuant, ParamFromRuntimePercentileScaling, PerTensorPoTScaling8bit, ActQuantSolver):
    """
    8-bit per-tensor unsigned int activations fixed-point quantizer with learned radix point
    initialized from runtime statistics.

    Examples:
        >>> from brevitas.nn import QuantReLU
        >>> act = QuantReLU(act_quant=Uint8ActPerTensorFixedPoint)
    """
    pass


class Int8BiasPerTensorFixedPointInternalScaling(
    IntQuant, MaxStatsScaling, PerTensorPoTScaling8bit, BiasQuantSolver):
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

