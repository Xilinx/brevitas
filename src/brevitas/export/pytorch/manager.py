from typing import Union, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from brevitas.quant_tensor import QuantTensor
from brevitas.export.manager import BaseManager, ExportContext
from brevitas.export.manager import _set_layer_export_handler, _set_layer_export_mode
from .handler.parameter import PytorchQuantConv2dHandler
from .handler.parameter import PytorchQuantConv1dHandler
from .handler.parameter import PytorchQuantLinearHandler
from .handler.act import PytorchQuantIdentityHandler
from .handler.act import PytorchQuantReLUHandler
from .handler.act import PytorchQuantHardTanhHandler
from .handler.pool import PytorchQuantMaxPool1d, PytorchQuantMaxPool2d
from .handler import qF


class PytorchQuantManager(BaseManager):
    target_name = 'torch'

    handlers = [
        PytorchQuantMaxPool1d,
        PytorchQuantMaxPool2d,
        PytorchQuantHardTanhHandler,
        PytorchQuantIdentityHandler,
        PytorchQuantReLUHandler,
        PytorchQuantConv1dHandler,
        PytorchQuantConv2dHandler,
        PytorchQuantLinearHandler]

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        _set_layer_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_layer_export_handler(cls, module)

    @classmethod
    def export(
            cls,
            module: Module,
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None):
        if qF is None:
            raise RuntimeError("torch.nn.quantized.functional cannot be imported.")
        if input_shape is None and input_t is None:
            raise RuntimeError("Export requires to pass in either input_shape or input_t")
        if input_shape is not None and input_t is not None:
            raise RuntimeError("Export accepts either an input shape or an input tensor, not both")
        if input_t is None and input_shape is not None:
            input_t = torch.empty(*input_shape)
        with ExportContext(cls):
            traced_module = cls.jit_inference_trace(module, input_t)
        if export_path is not None:
            traced_module.save(export_path)
        return traced_module