from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.export.onnx.handler import ONNXBaseHandler, QuantLSTMLayerHandler
from brevitas.export.common.handler.qcdq import (
    DQMixin,
    QCDQMixin,
    ZeroPointHandlerMixin,
    QCDQWeightQuantProxyHandlerMixin,
    QCDQDecoupledWeightQuantProxyHandlerMixin,
    QCDQActQuantProxyHandlerMixin,
    QCDQTruncQuantProxyHandlerMixin)
from brevitas.export.common import to_0dim_if_scalar
from brevitas.export.common.handler.base import QuantAxisMixin

from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn


class StdQCDQONNXQuantProxyHandler(
    ONNXBaseHandler, QCDQMixin, ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def clip_over_integers(self):
        return True

    @property
    def itemize_scalar_params(self):
        return False
    
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
    QCDQWeightQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXDecoupledWeightQuantProxyHandler(
    QCDQDecoupledWeightQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXActQuantProxyHandler(
    QCDQActQuantProxyHandlerMixin, StdQCDQONNXQuantProxyHandler):
    pass


class StdQCDQONNXBiasQuantProxyHandler(
    DQMixin, QuantAxisMixin, ZeroPointHandlerMixin, ONNXBaseHandler):
    handled_layer = BiasQuantProxyFromInjector
    
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
    
    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            int_biases = {
                tm.bias.data_ptr():
                    tm.quant_bias().int(float_datatype=False) for tm in module.tracked_module_list}
            self.symbolic_kwargs = {
                'int_biases': int_biases,
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width()}
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        int_bias = self.symbolic_kwargs['int_biases'][x.data_ptr()]
        scale = self.symbolic_kwargs['scale']
        bit_width = self.symbolic_kwargs['bit_width']
        zero_point = self.symbolic_kwargs['zero_point']
        assert scale is not None or input_scale is not None, 'Input scale required for bias export'
        assert bit_width is not None or input_bit_width is not None, 'Input bit width required for bias export'
        if input_scale is not None:
            scale = input_scale
        if input_bit_width is not None:
            bit_width = input_bit_width
        quant_axis = self.quant_axis(scale)
        scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(scale)
        zp = self.zero_point_with_dtype(True, bit_width, zp)  # assume signed is True
        y = self.dequantize_fn(
            int_bias.to(zp.dtype), scale, zero_point, quant_axis)
        return y, scale, zero_point, bit_width


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

