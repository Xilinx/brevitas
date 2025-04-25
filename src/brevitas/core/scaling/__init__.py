from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType

assert ScalingImplType
assert ScalingPerOutputType

from brevitas.core.stats import SCALAR_SHAPE

from .float_scaling import FloatScaling
from .int_scaling import IntScaling
from .int_scaling import PowerOfTwoIntScaling
from .pre_scaling import AccumulatorAwareParameterPreScaling
from .pre_scaling import AccumulatorAwareZeroCenterParameterPreScaling
from .pre_scaling import ParameterPreScalingWeightNorm
from .runtime import RuntimeStatsScaling
from .runtime import StatsFromParameterScaling
from .standalone import ConstScaling
from .standalone import ParameterFromRuntimeStatsScaling
from .standalone import ParameterFromStatsFromParameterScaling
from .standalone import ParameterScaling
from .standalone import TruncMsbScaling
from .standalone import TruncScalingWrapper

SCALING_STATS_REDUCE_DIM = 1
