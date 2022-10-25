from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector


from ..function import QuantizeLinearFn, DequantizeLinearFn, IntClipFn
from ..handler import StdONNXQuantLayerHandler, to_0dim_if_scalar


class QCDQQuantProxyHandler(StdONNXQuantLayerHandler, ABC):

    def validate(self, module):
        self.validate_8b_bit_width(module.bit_width(), le_then=True)
        assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'

    def quantize_symbolic_kwargs(cls, module):
        flat_scale = to_0dim_if_scalar(module.scale().flatten())
        zp =  to_0dim_if_scalar(module.zero_point().flatten()).expand_as(flat_scale) # Expand_as must go after 0-Dim check
        return {
            'scale': flat_scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed,zp),
            'dtype': torch.int8 if module.is_signed else torch.uint8,
            'axis': cls.quant_axis(module.scale())}

    def dequantize_symbolic_kwargs(cls, module):
        flat_scale = to_0dim_if_scalar(module.scale().flatten()) 
        zp = to_0dim_if_scalar(module.zero_point().flatten()).expand_as(flat_scale)
        return {
            'scale': flat_scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed, zp),
            'axis': cls.quant_axis(module.scale())}

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs['bit_width'] = module.bit_width()
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                module)
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.clip_symbolic_kwargs(
                module.is_narrow_range, module.is_signed, module.bit_width())
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                module)
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        clip_symbolic_kwargs = self.symbolic_kwargs['clip_symbolic_kwargs']
        dequantize_symbolic_kwargs = self.symbolic_kwargs['dequantize_symbolic_kwargs']
        scale = dequantize_symbolic_kwargs['scale']
        zero_point = dequantize_symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        x = QuantizeLinearFn.apply(x, *quantize_symbolic_kwargs.values(), bit_width)
        if clip_symbolic_kwargs is not None:
            x = IntClipFn.apply(x, *clip_symbolic_kwargs.values())
        x = DequantizeLinearFn.apply(x, *dequantize_symbolic_kwargs.values(), bit_width)
        return x, scale, zero_point, bit_width


class QCDQWeightQuantProxyHandler(QCDQQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector


class QCDQDecoupledWeightQuantProxyHandler(QCDQWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def quantize_symbolic_kwargs(cls, module):
        flat_scale = to_0dim_if_scalar(module.pre_scale().flatten())
        zp = to_0dim_if_scalar(module.pre_zero_point().flatten()).expand_as(flat_scale)
        return {
            'scale': flat_scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed, zp),
            'dtype': torch.int8 if module.is_signed else torch.uint8,
            'axis': cls.quant_axis(module.pre_scale())}

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        pre_scale = quantize_symbolic_kwargs['scale']
        pre_zero_point = quantize_symbolic_kwargs['zero_point']
        return out, pre_scale, pre_zero_point, scale, zero_point, bit_width


class QCDQActQuantProxyHandler(QCDQWeightQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector

class QCDQBiasQuantProxyHandler(StdONNXQuantLayerHandler):
    handled_layer = BiasQuantProxyFromInjector

    def validate(self, module):
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported'

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
        dtype = torch.int32 if int(bit_width.item()) > 8 else torch.int8
        flat_scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(flat_scale).to(dtype)
        y = DequantizeLinearFn.apply(
            int_bias.to(dtype), flat_scale, zp, self.quant_axis(scale))
        return y, scale, zero_point, bit_width


class QCDQTruncQuantProxyHandler(QCDQQuantProxyHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'output_bit_width': module.bit_width()}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor, signed: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        output_bit_width = self.symbolic_kwargs['output_bit_width']
        dtype = torch.int8 if signed else torch.uint8
        flat_scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(flat_scale)
        x = QuantizeLinearFn.apply(x, flat_scale, zp, dtype, self.quant_axis(scale))
        clip_symbolic_kwargs = self.clip_symbolic_kwargs(signed, False, output_bit_width)
        if clip_symbolic_kwargs is not None:
            x = IntClipFn.apply(x, *clip_symbolic_kwargs.values())
        x = DequantizeLinearFn.apply(x, scale.flatten(), zp, self.quant_axis(scale))
        return x, scale, zero_point, output_bit_width
