from abc import ABC
from functools import partial

import torch.nn.functional as F

from brevitas.export.onnx.vitis_ai import VitisAIManager
from brevitas.export.onnx.transform import move_domain_attributes_into_domain
from .handler import DPUQuantConv2dHandler, DPUQuantMaxPool2dHandler
from .handler import DPUQuantReLUHandler, DPUQuantEltwiseAddHandler
from .handler import DPUQuantAvgPool2dHandler, DPUQuantLinearHandler


def _handler_wrapper(handler, cached_io):
    handler = handler()
    handler.prepare_from_cached_io(cached_io)
    return handler


class PyXIRManager(VitisAIManager, ABC):
    target_name = 'PyXIR'

    model_transforms = [
        move_domain_attributes_into_domain]

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

    _cached_io_handler_map = {
        F.relu: partial(_handler_wrapper, DPUQuantReLUHandler),
        F.max_pool2d: partial(_handler_wrapper, DPUQuantMaxPool2dHandler)}


