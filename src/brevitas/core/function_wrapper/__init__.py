from .clamp import ClampMin, TensorClamp, ConstScalarClamp
from .misc import LogTwo, InplaceLogTwo, PowerOfTwo, Identity
from .ops_ste import CeilSte, ClampMinSte, FloorSte, RoundSte, RoundToZeroSte, TensorClampSte
from .shape import OverBatchOverOutputChannelView, OverBatchOverTensorView, OverOutputChannelView
from .shape import OverTensorView


class StatsInputViewShapeImpl(object):
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView
