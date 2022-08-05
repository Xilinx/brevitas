from typing import Tuple, Union, Optional
from torch.nn import Module
from torch import Tensor

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.quant_tensor import QuantTensor

from .handler import QCDQWeightQuantProxyHandler
from .handler import QCDQBiasQuantProxyHandler
from .handler import QCDQActQuantProxyHandler
from .handler import QCDQDecoupledWeightQuantProxyHandler
from .handler import QCDQTruncQuantProxyHandler

from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn
from .. import OPSET


class StdQCDQONNXManager(ONNXBaseManager):
    target_name = 'StdQCDQONNX'
    dequantize_tracing_input = False

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    handlers = [
        QCDQWeightQuantProxyHandler,
        QCDQBiasQuantProxyHandler,
        QCDQActQuantProxyHandler,
        QCDQDecoupledWeightQuantProxyHandler,
        QCDQTruncQuantProxyHandler
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

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            disable_warnings=True,
            **kwargs):
        output = super().export_onnx(
            module, input_shape, export_path, input_t,
            disable_warnings=disable_warnings, opset_version=OPSET, **kwargs)
        return output