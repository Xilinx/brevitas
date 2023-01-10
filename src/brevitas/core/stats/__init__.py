# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from brevitas.inject.enum import StatsOp
assert StatsOp

from .stats_op import AbsMax, AbsAve, AbsMaxAve, AbsPercentile, MeanLearnedSigmaStd, MeanSigmaStd
from .stats_op import AbsMinMax, NegativeMinOrZero, AbsMaxL2, PercentileInterval, NegativePercentileOrZero
from .stats_wrapper import SCALAR_SHAPE, DEFAULT_MOMENTUM
from .stats_wrapper import _RuntimeStats, _ParameterListStats, _Stats