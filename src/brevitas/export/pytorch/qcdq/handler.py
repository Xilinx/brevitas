from abc import ABC

import torch
from torch import Tensor

from brevitas.function.ops import tensor_clamp_
from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.common.handler.qcdq import (
    QCDQMixin,
    QCDQQuantProxyHandlerMixin,
    QCDQWeightQuantProxyHandlerMixin,
    QCDQBiasQuantProxyHandlerMixin,
    QCDQActQuantProxyHandlerMixin,
    QCDQTruncQuantProxyHandlerMixin)


class TorchQCDQQuantProxyHandler(
    BaseHandler, QCDQMixin, ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.symbolic_kwargs = {}

    @property
    def clip_over_integers(self):
        return True
    
    @property
    def flatten_dequantize_params(self):
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
            y = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        else:
            y = torch.quantize_per_channel(x, scale, zero_point, axis, dtype)
        return y.int_repr()
    
    def clip_fn(self, x, min_val, max_val):
        return torch.clip(x, min_val, max_val)
    
    def dequantize_fn(self, x, scale, zero_point, axis, bit_width=None):
        return (x - zero_point) * scale
    
    def forward(self, *args, **kwargs):
        return self.symbolic_execution(*args, **kwargs)



class TorchQCDQWeightQuantProxyHandler(
    QCDQWeightQuantProxyHandlerMixin, TorchQCDQQuantProxyHandler):
    pass


class TorchQCDQActQuantProxyHandler(
    QCDQActQuantProxyHandlerMixin, TorchQCDQQuantProxyHandler):
    pass


class TorchQCDQBiasQuantProxyHandler(
    QCDQBiasQuantProxyHandlerMixin, BaseHandler):

    @classmethod    
    def int8_dtype(cls):
        return torch.qint8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.qint32
    
    @property
    def flatten_dequantize_params(self):
        return False
    
    def validate(self, module):
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return x.dequantize()
    
    def forward(self, *args, **kwargs):
        return self.symbolic_execution(*args, **kwargs)


class TorchQCDQTruncQuantProxyHandler(
   TorchQCDQQuantProxyHandler, QCDQTruncQuantProxyHandlerMixin):
    pass