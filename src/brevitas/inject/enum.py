# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from enum import auto

from brevitas.utils.python_utils import AutoName


class BitWidthImplType(AutoName):
    """

    """
    CONST = auto()
    PARAMETER = auto()
    STATEFUL_CONST = auto()


class QuantType(AutoName):
    """

    """
    BINARY = auto()
    TERNARY = auto()
    INT = auto()
    FP = auto()


class RestrictValueType(AutoName):
    """

    """
    FP = auto()
    LOG_FP = auto()
    INT = auto()
    POWER_OF_TWO = auto()


class FloatToIntImplType(AutoName):
    """

    """
    ROUND = auto()
    CEIL = auto()
    FLOOR = auto()
    ROUND_TO_ZERO = auto()
    DPU = auto()
    LEARNED_ROUND = auto()


class LearnedRoundImplType(AutoName):
    """
    """
    HARD_SIGMOID = auto()
    SIGMOID = auto()


class ScalingImplType(AutoName):
    """

    """
    HE = auto()
    CONST = auto()
    STATS = auto()
    AFFINE_STATS = auto()
    PARAMETER = auto()
    PARAMETER_FROM_STATS = auto()


class StatsOp(AutoName):
    """

    """
    # One sided statistics over absolute value
    # Typically adopted for symmetric quantization
    MAX = auto()
    AVE = auto()
    MAX_AVE = auto()
    MEAN_SIGMA_STD = auto()
    MEAN_LEARN_SIGMA_STD = auto()
    PERCENTILE = auto()
    # Two sided statistics
    # Typically adopted for asymmetric quantization
    MIN_MAX = auto()
    PERCENTILE_INTERVAL = auto()
