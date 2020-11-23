from brevitas.inject.enum import StatsOp
assert StatsOp

from .stats_op import AbsMax, AbsAve, AbsMaxAve, AbsPercentile, MeanLearnedSigmaStd, MeanSigmaStd
from .stats_op import AbsMinMax, NegativeMinOrZero
from .stats_wrapper import SCALAR_SHAPE, DEFAULT_MOMENTUM
from .stats_wrapper import _RuntimeStats, _ParameterListStats, _Stats