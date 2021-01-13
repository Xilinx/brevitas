from typing import Tuple, Optional, Union
from packaging import version

from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor
from brevitas.export.onnx.base import ONNXBaseManager

from .handler.parameter import StdONNXQuantConv2dHandler
from .handler.parameter import StdONNXQuantConv1dHandler
from .handler.parameter import StdONNXQuantLinearHandler
from .handler.act import StdONNXQuantReLUHandler
from .handler.act import StdONNXQuantHardTanhHandler
from .handler.act import StdONNXQuantIdentityHandler
from .handler.act import StdONNXQuantTanhHandler
from .handler.act import StdONNXQuantSigmoidHandler
from .handler.pool import StdONNXQuantMaxPool1d
from .handler.pool import StdONNXQuantMaxPool2d
from . import OPSET


class StdONNXManager(ONNXBaseManager):

    handlers = [
        StdONNXQuantConv1dHandler,
        StdONNXQuantConv2dHandler,
        StdONNXQuantLinearHandler,
        StdONNXQuantReLUHandler,
        StdONNXQuantHardTanhHandler,
        StdONNXQuantIdentityHandler,
        StdONNXQuantTanhHandler,
        StdONNXQuantSigmoidHandler,
        StdONNXQuantMaxPool1d,
        StdONNXQuantMaxPool2d]

    onnx_passes = [
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    @classmethod
    def solve_enable_onnx_checker(cls, export_kwargs):
        if torch_version >= version.parse('1.5.0'):
            export_kwargs['enable_onnx_checker'] = True

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