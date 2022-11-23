from torch.nn import Module

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode

from .handler import (
    StdQCDQONNXWeightQuantProxyHandler,
    StdQCDQONNXBiasQuantProxyHandler,
    StdQCDQONNXActQuantProxyHandler,
    StdQCDQONNXDecoupledWeightQuantProxyHandler,
    StdQCDQONNXTruncQuantProxyHandler)

from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn
from ..manager import StdONNXBaseManager


class StdQCDQONNXManager(StdONNXBaseManager):
    target_name = 'StdQCDQONNX'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        StdQCDQONNXWeightQuantProxyHandler,
        StdQCDQONNXBiasQuantProxyHandler,
        StdQCDQONNXActQuantProxyHandler,
        StdQCDQONNXDecoupledWeightQuantProxyHandler,
        StdQCDQONNXTruncQuantProxyHandler
    ]

    custom_fns = [
        DebugMarkerFunction,
        QuantizeLinearFn,
        DequantizeLinearFn,
        IntClipFn
    ]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)