from brevitas.inject.enum import QuantType

from brevitas.quant.solver import WeightQuantSolver
from brevitas.quant.solver import BiasQuantSolver
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import TruncQuantSolver
from brevitas.quant.solver import ClampQuantSolver

__all__ = [
    'NoneWeightQuant',
    'NoneActQuant',
    'NoneBiasQuant',
    'NoneTruncQuant',
    'NoneClampQuant'
]


class NoneWeightQuant(WeightQuantSolver):
    """
    Base quantizer used when weight_quant=None.
    """
    quant_type = QuantType.FP


class NoneActQuant(ActQuantSolver):
    """
    Base quantizer used when act_quant=None or input_quant=None or output_quant=None.
    """
    quant_type = QuantType.FP


class NoneBiasQuant(BiasQuantSolver):
    """
    Base quantizer used when bias_quant=None.
    """
    quant_type = QuantType.FP
    requires_input_scale = False
    requires_input_bit_width = False


class NoneTruncQuant(TruncQuantSolver):
    """
    Base quantizer used when trunc_quant=None.
    """
    quant_type = QuantType.FP


class NoneClampQuant(ClampQuantSolver):
    """
    Base quantizer used when clamp_quant=None.
    """
    quant_type = QuantType.FP