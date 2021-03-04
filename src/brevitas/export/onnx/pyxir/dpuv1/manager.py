from .handler import DPUv1QuantConv2dHandler, DPUv1QuantMaxPool2dHandler
from ..handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from ..handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler
from ..manager import PyXIRManager


class DPUv1Manager(PyXIRManager):
    target_name = 'PyXIR+DPUv1'

    handlers = [
        DPUQuantReLUHandler,
        DPUQuantEltwiseAddHandler,
        DPUQuantAvgPool2dHandler,
        DPUQuantLinearHandler,
        DPUv1QuantConv2dHandler,
        DPUv1QuantMaxPool2dHandler]
