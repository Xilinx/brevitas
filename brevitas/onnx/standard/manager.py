from typing import Tuple, Optional, Union

from torch import Tensor
from torch.nn import Module

from brevitas.quant_tensor import QuantTensor
from brevitas.onnx.base import BaseManager

from .handler.parameter import ONNXQuantConv2dHandler
from .handler.parameter import ONNXQuantConv1dHandler
from .handler.parameter import ONNXQuantLinearHandler
from .handler.act import ONNXQuantReLUHandler
from .handler.act import ONNXQuantHardTanhHandler
from .handler.act import ONNXQuantIdentityHandler
from .handler.act import ONNXQuantTanhHandler
from .handler.act import ONNXQuantSigmoidHandler
from .handler.pool import ONNXQuantMaxPool1d
from .handler.pool import ONNXQuantMaxPool2d
from . import OPSET


class StandardONNXManager(BaseManager):

    handlers = [
        ONNXQuantConv1dHandler,
        ONNXQuantConv2dHandler,
        ONNXQuantLinearHandler,
        ONNXQuantReLUHandler,
        ONNXQuantHardTanhHandler,
        ONNXQuantIdentityHandler,
        ONNXQuantTanhHandler,
        ONNXQuantSigmoidHandler,
        ONNXQuantMaxPool1d,
        ONNXQuantMaxPool2d]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...],
            export_path: str,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        return super().export_onnx(
            module, input_shape, export_path, input_t, opset_version=OPSET, **kwargs)