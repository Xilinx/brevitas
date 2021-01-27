from brevitas.inject.enum import ScalingImplType
assert ScalingImplType

from .int_scaling import IntScaling, PowerOfTwoIntScaling
from .runtime import StatsFromParameterScaling, RuntimeStatsScaling
from .standalone import ParameterFromRuntimeStatsScaling, ParameterScaling, ConstScaling
from brevitas.core.stats import SCALAR_SHAPE

SCALING_STATS_REDUCE_DIM = 1
