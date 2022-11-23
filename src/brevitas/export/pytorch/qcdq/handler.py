from abc import ABC

import torch
from torch import Tensor

from brevitas.function.ops import tensor_clamp_
from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.common.handler.qcdq import (
    QCDQQuantProxyHandlerMixin,
    QCDQWeightQuantProxyHandlerMixin,
    QCDQBiasQuantProxyHandlerMixin,
    QCDQActQuantProxyHandlerMixin,
    QCDQTruncQuantProxyHandlerMixin)


class TorchQCDQQuantProxyHandler(
    BaseHandler, QCDQQuantProxyHandlerMixin, ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.symbolic_kwargs = {}

    @property
    def clip_over_integers(self):
        return False

    @classmethod    
    def int8_dtype(cls):
        return torch.qint8

    @classmethod    
    def uint8_dtype(cls):
        return torch.quint8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.qint32

    def validate(self, module):
        self.validate_8b_bit_width(module.bit_width(), le_then=True)
        assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'
        
    def quantize_fn(self, x, scale, zero_point, dtype, axis, bit_width=None):
        if axis is None:
            return torch.quantize_per_tensor(x, scale, zero_point, dtype)
        else:
            return torch.quantize_per_channel(x, scale, zero_point, axis, dtype)
    
    def clip_fn(self, x, min_val, max_val):
        y = torch.where(x > max_val, max_val, x)
        return torch.where(y < min_val, min_val, y)
    
    def dequantize_fn(self, x, scale, zero_point, axis, bit_width=None):
        return x.dequantize()
    
    def forward(self, x):
        return self.symbolic_execution(x)


class TorchQCDQWeightQuantProxyHandler(
    TorchQCDQQuantProxyHandler, QCDQWeightQuantProxyHandlerMixin):
    pass


class TorchQCDQActQuantProxyHandler(
    TorchQCDQQuantProxyHandler, QCDQActQuantProxyHandlerMixin):
    pass


class TorchQCDQBiasQuantProxyHandler(
    BaseHandler, QCDQBiasQuantProxyHandlerMixin):

    @classmethod    
    def int8_dtype(cls):
        return torch.qint8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.qint32
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return x.dequantize()


class TorchQCDQTruncQuantProxyHandler(
   TorchQCDQQuantProxyHandler, QCDQTruncQuantProxyHandlerMixin):
    pass