from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.common.handler.qcdq import (
    QCDQMixin,
    QCDQWeightQuantProxyHandlerMixin,
    QCDQDecoupledWeightQuantProxyHandlerMixin,
    QCDQBiasQuantProxyHandlerMixin,
    QCDQActQuantProxyHandlerMixin,
    QCDQTruncQuantProxyHandlerMixin)

from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn


class StdQCDQONNXQuantProxyHandler(
    ONNXBaseHandler, QCDQMixin, ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def clip_over_integers(self):
        return True
    
    @classmethod    
    def int8_dtype(cls):
        return torch.int8

    @classmethod    
    def uint8_dtype(cls):
        return torch.uint8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.int32

    def validate(self, module):
        self.validate_8b_bit_width(module.bit_width(), le_then=True)
        assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'

    def quantize_fn(self, x, scale, zero_point, dtype, axis, bit_width=None):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis, bit_width)
    
    def clip_fn(self, x, min_val, max_val):
        return IntClipFn.apply(x, min_val, max_val)
    
    def dequantize_fn(self, x, scale, zero_point, axis, bit_width=None):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis, bit_width)


class StdQCDQONNXWeightQuantProxyHandler(
    StdQCDQONNXQuantProxyHandler, QCDQWeightQuantProxyHandlerMixin):
    pass


class StdQCDQONNXDecoupledWeightQuantProxyHandler(
    StdQCDQONNXQuantProxyHandler, QCDQDecoupledWeightQuantProxyHandlerMixin):
    pass


class StdQCDQONNXActQuantProxyHandler(
    StdQCDQONNXQuantProxyHandler, QCDQActQuantProxyHandlerMixin):
    pass

class StdQCDQONNXBiasQuantProxyHandler(
    ONNXBaseHandler, QCDQBiasQuantProxyHandlerMixin):
    
    def validate(self, module):
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'
        
    @classmethod    
    def int8_dtype(cls):
        return torch.int8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.int32
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis)


class StdQCDQONNXTruncQuantProxyHandler(
    StdQCDQONNXQuantProxyHandler, QCDQTruncQuantProxyHandlerMixin):
    pass

