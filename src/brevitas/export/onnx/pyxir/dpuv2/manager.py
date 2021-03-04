from .handler import DPUv2QuantConv2dHandler, DPUv2QuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler
from ..manager import PyXIRManager


class DPUv2Manager(PyXIRManager):
    target_name = 'PyXIR+DPUv2'

    handlers = [
        DPUQuantReLUHandler,
        DPUQuantEltwiseAddHandler,
        DPUQuantAvgPool2dHandler,
        DPUQuantLinearHandler,
        DPUv2QuantConv2dHandler,
        DPUv2QuantMaxPool2dHandler]