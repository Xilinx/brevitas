from typing import Tuple, Optional, Union
from packaging import version

from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor
from brevitas.export.onnx.manager import ONNXBaseManager
from brevitas.export.manager import _set_layer_export_handler, _set_layer_export_mode

from .handler.base import StdQOpONNXQuantLayerHandler
from .handler.parameter import StdQOpONNXQuantConv2dHandler
from .handler.parameter import StdQOpONNXQuantConv1dHandler
from .handler.parameter import StdQOpONNXQuantLinearHandler
from .handler.act import StdQOpONNXQuantReLUHandler
from .handler.act import StdQOpONNXQuantHardTanhHandler
from .handler.act import StdQOpONNXQuantIdentityHandler
from .handler.act import StdQOpONNXQuantTanhHandler
from .handler.act import StdQOpONNXQuantSigmoidHandler
from .handler.pool import StdQOpONNXQuantMaxPool1d
from .handler.pool import StdQOpONNXQuantMaxPool2d
from .. import OPSET
from ..function import QuantizeLinearFn, DequantizeLinearFn


class StdQOpONNXManager(ONNXBaseManager):
    target_name = 'StdQOpONNX'

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
        StdQOpONNXQuantConv1dHandler,
        StdQOpONNXQuantConv2dHandler,
        StdQOpONNXQuantLinearHandler,
        StdQOpONNXQuantReLUHandler,
        StdQOpONNXQuantHardTanhHandler,
        StdQOpONNXQuantIdentityHandler,
        StdQOpONNXQuantTanhHandler,
        StdQOpONNXQuantSigmoidHandler,
        StdQOpONNXQuantMaxPool1d,
        StdQOpONNXQuantMaxPool2d]

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
                deq_kwargs = StdQOpONNXQuantLayerHandler.dequant_symbolic_kwargs_from_cached_io(
                    cached_inp)
                input = DequantizeLinearFn.apply(input, *deq_kwargs.values())
            output = fn(input, *args, **kwargs)
            if cached_out is not None:
                q_kwargs = StdQOpONNXQuantLayerHandler.quant_symbolic_kwargs_from_cached_io(
                    cached_out)
                output = QuantizeLinearFn.apply(output, *q_kwargs.values())
        else:
            output = fn(input, *args, **kwargs)
        return output

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        _set_layer_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_layer_export_handler(cls, module)

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