from functools import partial

from torch.nn import functional as F

from .handler import DPUv2QuantConv2dHandler, DPUv2QuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler
from ..manager import PyXIRManager, _handler_wrapper


class DPUv2Manager(PyXIRManager):
    target_name = 'PyXIR+DPUv2'

    handlers = [
        DPUQuantReLUHandler,
        DPUQuantEltwiseAddHandler,
        DPUQuantAvgPool2dHandler,
        DPUQuantLinearHandler,
        DPUv2QuantConv2dHandler,
        DPUv2QuantMaxPool2dHandler]

    _cached_io_handler_map = {
    F.relu: partial(_handler_wrapper, DPUQuantReLUHandler),
    F.max_pool2d: partial(_handler_wrapper, DPUv2QuantMaxPool2dHandler)}