from typing import Tuple, Optional, Union
from packaging import version

from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor
from brevitas.export.onnx.base import ONNXBaseManager

from .function import QuantizeLinearFunction, DequantizeLinearFunction
from .handler.base import StdONNXQuantLayerHandler
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
    target_name = 'StdONNX'

    _fn_to_cache = [
        F.relu,
        F.relu6,
        F.hardtanh,
        F.max_pool1d,
        F.max_pool2d,
        F.max_pool3d,
        F.adaptive_max_pool1d,
        F.adaptive_max_pool2d,
        F.adaptive_max_pool3d,
    ]

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
    def _trace_fn_dispatcher(cls, fn, input, *args, **kwargs):
        cached_io = cls._fn_cache.pop(0)
        if cached_io is not None:
            cached_inp, cached_out = cached_io
            if cached_inp is not None:
                deq_kwargs = StdONNXQuantLayerHandler.dequant_symbolic_kwargs_from_cached_io(
                    cached_inp)
                input = DequantizeLinearFunction.apply(input, *deq_kwargs.values())
            output = fn(input, *args, **kwargs)
            if cached_out is not None:
                q_kwargs = StdONNXQuantLayerHandler.quant_symbolic_kwargs_from_cached_io(cached_out)
                output = QuantizeLinearFunction.apply(output, *q_kwargs.values())
        else:
            output = fn(input, *args, **kwargs)
        return output

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...],
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        output = super().export_onnx(
            module, input_shape, export_path, input_t, opset_version=OPSET, **kwargs)
        return output