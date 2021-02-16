from brevitas.quant.base import *
from brevitas.quant.solver.weight import WeightQuantSolver
from brevitas.quant.solver.bias import BiasQuantSolver
from brevitas.quant.solver.act import ActQuantSolver


__all__ = [
    'Int8WeightPerTensorFixedPoint',
    'Int8ActPerTensorFixedPoint',
    'Uint8ActPerTensorFixedPoint',
    'Int8BiasPerTensorFixedPoint'
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


class Int8BiasPerTensorFixedPoint(
    IntQuant, MaxStatsScaling, PerTensorPoTScaling8bit, BiasQuantSolver):
    """
    8-bit per-tensor signed fixed-point bias quantizer with the radix point computed
    from backpropagated statistics of the bias tensor.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=True, bias_quant=Int8BiasPerTensorFixedPoint)
    """
    requires_input_scale = False
    requires_input_bit_width = False

