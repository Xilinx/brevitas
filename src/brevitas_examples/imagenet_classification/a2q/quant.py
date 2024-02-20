# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.quant import Int8AccumulatorAwareWeightQuant
from brevitas.quant import Int8AccumulatorAwareZeroCenterWeightQuant

__all__ = ["CommonIntAccumulatorAwareWeightQuant", "CommonIntAccumulatorAwareZeroCenterWeightQuant"]

SCALING_MIN_VAL = 1e-8


class CommonIntAccumulatorAwareWeightQuant(Int8AccumulatorAwareWeightQuant):
    """A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance"""
    bit_width = None
    scaling_min_val = SCALING_MIN_VAL
    pre_scaling_min_val = SCALING_MIN_VAL


class CommonIntAccumulatorAwareZeroCenterWeightQuant(Int8AccumulatorAwareZeroCenterWeightQuant):
    """A2Q+: Improving Accumulator-Aware Weight Quantization"""
    bit_width = None
    scaling_min_val = SCALING_MIN_VAL
    pre_scaling_min_val = SCALING_MIN_VAL
