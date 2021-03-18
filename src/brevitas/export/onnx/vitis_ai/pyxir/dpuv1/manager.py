from torch.nn import functional as F
from functools import partial

from .handler import DPUv1QuantConv2dHandler, DPUv1QuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler
from ..manager import PyXIRManager, _handler_wrapper


class DPUv1Manager(PyXIRManager):
    target_name = 'PyXIR+DPUv1'

    handlers = [
        DPUQuantReLUHandler,
        DPUQuantEltwiseAddHandler,
        DPUQuantAvgPool2dHandler,
        DPUQuantLinearHandler,
        DPUv1QuantConv2dHandler,
        DPUv1QuantMaxPool2dHandler]

    _cached_io_handler_map = {
    F.relu: partial(_handler_wrapper, DPUQuantReLUHandler),
    F.max_pool2d: partial(_handler_wrapper, DPUv1QuantMaxPool2dHandler)}