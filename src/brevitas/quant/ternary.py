# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper import InplaceTensorClampSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import NarrowIntQuant
from brevitas.quant.base import PerTensorConstScaling2bit
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver

__all__ = [
    'SignedTernaryWeightPerTensorConst',
    'SignedTernaryActPerTensorConst',]


class SignedTernaryWeightPerTensorConst(NarrowIntQuant,
                                        PerTensorConstScaling2bit,
                                        WeightQuantSolver):
    """
    Signed ternary weight quantizer with constant scale factor and inplace clipping to the scale.

    Examples:
        >>> from brevitas.nn import QuantLinear
        >>> fc = QuantLinear(10, 5, bias=False, weight_quant=SignedTernaryWeightPerTensorConst)
        >>> fc.quant_weight()
    """
    tensor_clamp_impl = InplaceTensorClampSte
    scaling_const = 0.1


class SignedTernaryActPerTensorConst(NarrowIntQuant, PerTensorConstScaling2bit, ActQuantSolver):
    """
    Examples:
        >>> from brevitas.nn import QuantIdentity
        >>> act = QuantIdentity(act_quant=SignedTernaryActPerTensorConst)
    """
    tensor_clamp_impl = TensorClamp
    min_val = -1.0
    max_val = 1.0
