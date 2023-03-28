# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.inject.enum import StatsOp

assert StatsOp

from .stats_op import AbsAve
from .stats_op import AbsMax
from .stats_op import AbsMaxAve
from .stats_op import AbsMaxL2
from .stats_op import AbsMinMax
from .stats_op import AbsPercentile
from .stats_op import L1Norm
from .stats_op import L2Norm
from .stats_op import MeanLearnedSigmaStd
from .stats_op import MeanSigmaStd
from .stats_op import NegativeMinOrZero
from .stats_op import NegativePercentileOrZero
from .stats_op import PercentileInterval
from .stats_wrapper import _ParameterListStats
from .stats_wrapper import _RuntimeStats
from .stats_wrapper import _Stats
from .stats_wrapper import DEFAULT_MOMENTUM
from .stats_wrapper import SCALAR_SHAPE
