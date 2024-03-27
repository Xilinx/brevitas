# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.quant.base import MaxStatsScaling
from brevitas.quant.base import PerTensorFloatScaling8bit
from brevitas.quant.experimental.float import Fp8e4m3Weight
from brevitas.quant.solver.weight import WeightQuantSolver

__all__ = ['Fp8e4m3OCPWeightPerTensorFloat']


class Fp8e4m3OCPWeightPerTensorFloat(Fp8e4m3Weight,
                                     MaxStatsScaling,
                                     PerTensorFloatScaling8bit,
                                     WeightQuantSolver):
    pass
