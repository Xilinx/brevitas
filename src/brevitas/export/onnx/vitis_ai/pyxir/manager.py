from abc import ABC
from functools import partial

import torch.nn.functional as F

from brevitas.export.onnx.vitis_ai import VitisAIManager
from .handler import DPUQuantConv2dHandler, DPUQuantMaxPool2dHandler
from .handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from .handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler
from .function import DPUQuantLinearFn, DPUQuantReLUFn, DPUQuantConv2dFn
from .function import DPUQuantAvgPoolFn, DPUQuantMaxPoolFn, DPUQuantEltwiseAddFn


def _handler_wrapper(handler, cached_io):
    handler = handler()
    handler.prepare_from_cached_io(cached_io)
    return handler


class PyXIRManager(VitisAIManager, ABC):
    target_name = 'PyXIR'

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        DPUQuantReLUHandler,
        DPUQuantEltwiseAddHandler,
        DPUQuantAvgPool2dHandler,
        DPUQuantLinearHandler,
        DPUQuantConv2dHandler,
        DPUQuantMaxPool2dHandler]

    custom_fns = [
        DPUQuantEltwiseAddFn,
        DPUQuantMaxPoolFn,
        DPUQuantConv2dFn,
        DPUQuantReLUFn,
        DPUQuantLinearFn,
        DPUQuantAvgPoolFn,
    ]

    _cached_io_handler_map = {
        F.relu: partial(_handler_wrapper, DPUQuantReLUHandler),
        F.max_pool2d: partial(_handler_wrapper, DPUQuantMaxPool2dHandler)}


