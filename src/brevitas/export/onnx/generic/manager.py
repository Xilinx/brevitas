from typing import Tuple, Union, Optional
from torch.nn import Module
from torch import Tensor

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.export.manager import _set_proxy_export_handler, _set_proxy_export_mode

from .handler import BrevitasActQuantProxyHandler
from .handler import BrevitasBiasQuantProxyHandler
from .handler import BrevitasWeightQuantProxyHandler
from .handler import BrevitasTruncQuantProxyHandler
from .handler import BrevitasDecoupledWeightQuantProxyHandler

from .function import BrevitasQuantFn
from .function import BrevitasTruncFn
from .function import BrevitasBinaryQuantFn


class BrevitasONNXManager(ONNXBaseManager):
    target_name = 'brevitas'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        BrevitasActQuantProxyHandler,
        BrevitasBiasQuantProxyHandler,
        BrevitasWeightQuantProxyHandler,
        BrevitasDecoupledWeightQuantProxyHandler,
        BrevitasTruncQuantProxyHandler
    ]

    custom_fns = [
        DebugMarkerFunction,
        BrevitasQuantFn,
        BrevitasBinaryQuantFn,
        BrevitasTruncFn
    ]

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        # proxy level export
        _set_proxy_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        # proxy level export
        _set_proxy_export_handler(cls, module)