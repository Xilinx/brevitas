from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler, QuantLSTMLayerHandler
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
    
    @property
    def flatten_dequantize_params(self):
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

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis)
    
    def clip_fn(self, x, min_val, max_val):
        return IntClipFn.apply(x, min_val, max_val)
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis)


class StdQCDQONNXWeightQuantProxyHandler(
    QCDQWeightQuantProxyHandlerMixin,StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXDecoupledWeightQuantProxyHandler(
    QCDQDecoupledWeightQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXActQuantProxyHandler(
    QCDQActQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXBiasQuantProxyHandler(
    QCDQBiasQuantProxyHandlerMixin, ONNXBaseHandler):
    
    def validate(self, module):
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'
        
    @classmethod    
    def int8_dtype(cls):
        return torch.int8
    
    @classmethod    
    def int32_dtype(cls):
        return torch.int32
    
    @property
    def flatten_dequantize_params(self):
        return True
    
    def quantize_fn(self, x, scale, zero_point, qdtype, axis):
        dtype = torch.int32 if qdtype==torch.qint32 else torch.int8
        return x.int(float_datatype=False).type(dtype)
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis)


class StdQCDQONNXTruncQuantProxyHandler(
    QCDQTruncQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXQuantLSTMLayerHandler(QuantLSTMLayerHandler):
    
    def quantized_cell_symbolic_execution(
        self,
        quant_input, 
        quant_hidden_state, 
        quant_cell_state, 
        quant_weight_ii,
        quant_weight_if, 
        quant_weight_ic, 
        quant_weight_io, 
        quant_weight_hi,
        quant_weight_hf, 
        quant_weight_hc, 
        quant_weight_ho, 
        quant_bias_input,
        quant_bias_forget,
        quant_bias_cell,
        quant_bias_output):
        raise RuntimeError(
            "Quantized LSTM cell is not supported for ONNX QCDQ " 
            "(weights only quantization is). Use export_qonnx.")

