from typing import Union

from torch import Tensor
from torch.nn import Module

from brevitas.quant_tensor import QuantTensor
from brevitas.export.base import BaseManager
from .handler.parameter import PytorchQuantConv2dHandler
from .handler.parameter import PytorchQuantConv1dHandler
from .handler.parameter import PytorchQuantLinearHandler
from .handler.act import PytorchQuantIdentityHandler
from .handler.act import PytorchQuantReLUHandler
from .handler.act import PytorchQuantHardTanhHandler
from .handler.pool import PytorchQuantMaxPool1d, PytorchQuantMaxPool2d


class PytorchQuantManager(BaseManager):

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
    def export(cls, module: Module, input_t: Union[Tensor, QuantTensor]):
        traced_module = cls.jit_trace(module, input_t)
        return traced_module