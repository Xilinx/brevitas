from abc import ABC

import torch

from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.common.handler.qcdq import (
    QCDQMixin,
    QCDQWeightQuantProxyHandlerMixin,
    QCDQQuantProxyHandlerMixin,
    QCDQActQuantProxyHandlerMixin,
    QCDQTruncQuantProxyHandlerMixin)
from brevitas.export.common import to_0dim_if_scalar, to_item_if_0dim


def _itemize_clip_bounds(clip_args):
    if clip_args is not None:
        clip_args['min_val'] = clip_args['min_val'].item()
        clip_args['max_val'] = clip_args['max_val'].item()
    return clip_args


class TorchQCDQQuantProxyHandler(
    BaseHandler, QCDQMixin, ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.symbolic_kwargs = {}

    @property
    def clip_over_integers(self):
        return False
    
    @property
    def itemize_scalar_params(self):
        return True

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
        assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'
        
    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        if axis is None:
            y = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        else:
            y = torch.quantize_per_channel(x, scale, zero_point, axis, dtype)
        return y
    
    def clip_fn(self, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)
    
    def dequantize_fn(self, x, scale, zero_point, axis):
        return x.dequantize()
    
    def forward(self, *args, **kwargs):
        return self.symbolic_execution(*args, **kwargs)



class TorchQCDQWeightQuantProxyHandler(
    QCDQWeightQuantProxyHandlerMixin, TorchQCDQQuantProxyHandler):
    
    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)

    
class TorchQCDQActQuantProxyHandler(
    QCDQActQuantProxyHandlerMixin, TorchQCDQQuantProxyHandler):
    
    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQBiasQuantProxyHandler(
    QCDQQuantProxyHandlerMixin, TorchQCDQQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector
    
    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)
    
    def validate(self, module):
        assert module.is_signed, "Unsigned bias not supported."
        return super().validate(module)
    
    def symbolic_execution(self, x, input_scale=None, input_bit_width=None):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        # update symbolic kwargs based on whether they are None
        quant_scale = self.symbolic_kwargs['quantize_symbolic_kwargs']['scale']
        dequant_scale = self.symbolic_kwargs['dequantize_symbolic_kwargs']['scale']
        quant_zp = self.symbolic_kwargs['quantize_symbolic_kwargs']['zero_point']
        dequant_zp = self.symbolic_kwargs['dequantize_symbolic_kwargs']['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        if quant_scale is None and dequant_scale is None and bit_width is None:
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                input_scale, quant_zp, input_bit_width, is_signed=True)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                input_scale, dequant_zp, input_bit_width, is_signed=True)
        elif quant_scale is not None and dequant_scale is not None and bit_width is None:
            quant_scale = self.symbolic_kwargs['quantize_symbolic_kwargs']['scale']
            dequant_scale = self.symbolic_kwargs['dequantize_symbolic_kwargs']['scale']
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                quant_scale, quant_zp, input_bit_width, is_signed=True)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                dequant_scale, dequant_zp, input_bit_width, is_signed=True)
        elif quant_scale is None and dequant_scale is None and bit_width is not None:
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                input_scale, quant_zp, bit_width, is_signed=True)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                input_scale, dequant_zp, bit_width, is_signed=True)
        return super(TorchQCDQBiasQuantProxyHandler, self).symbolic_execution(x)


class TorchQCDQTruncQuantProxyHandler(
    TorchQCDQQuantProxyHandler, QCDQTruncQuantProxyHandlerMixin):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)