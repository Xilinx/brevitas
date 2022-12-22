from abc import abstractmethod, ABC

import torch
from torch import Tensor

from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.export.common import to_0dim_if_scalar
from .base import QuantAxisMixin, ClipMixin, ZeroPointHandlerMixin, BitWidthHandlerMixin


class DQMixin(ABC):
    
    @abstractmethod
    def dequantize_fn(self, x, scale, zero_point, axis):
        pass
    
    @property
    @abstractmethod
    def flatten_dequantize_params(self):
        pass
    
    def assert_ge_zero(self, *args):
        for a in args:
            assert a >= 0.
    
    
class QCDQMixin(DQMixin):
    
    @classmethod    
    @abstractmethod
    def uint8_dtype(cls):
        pass
    
    @classmethod    
    @abstractmethod
    def int8_dtype(cls):
        pass
    
    @classmethod    
    @abstractmethod
    def int32_dtype(cls):
        pass
    
    @property
    @abstractmethod
    def clip_over_integers(self):
        pass
    
    @abstractmethod
    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        pass
    
    @abstractmethod
    def clip_fn(self, x, min_val, max_val):
        pass
    
    @abstractmethod
    def validate(self):
        pass


class QCDQQuantProxyHandlerMixin(
    QuantAxisMixin, ClipMixin, ZeroPointHandlerMixin, BitWidthHandlerMixin, QCDQMixin, ABC):

    def quantize_symbolic_kwargs(cls, module):
        flat_scale = to_0dim_if_scalar(module.scale().flatten())
        # expand_as must go after 0-Dim check
        zp =  to_0dim_if_scalar(module.zero_point().flatten()).expand_as(flat_scale) 
        return {
            'scale': flat_scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed,zp),
            'dtype': cls.int8_dtype() if module.is_signed else cls.uint8_dtype(),
            'axis': cls.quant_axis(module.scale())}

    def dequantize_symbolic_kwargs(cls, module):
        quant_axis = cls.quant_axis(module.scale())
        if cls.flatten_dequantize_params:
            scale = to_0dim_if_scalar(module.scale().flatten()) 
            zp = to_0dim_if_scalar(module.zero_point().flatten()).expand_as(scale)
        else:
            scale = module.scale()
            zp = module.zero_point().expand_as(scale)
        return {
            'scale': scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed, zp),
            'axis': quant_axis}

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs['bit_width'] = module.bit_width()
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                module)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                module)
            if self.clip_over_integers:
                self.symbolic_kwargs['clip_symbolic_kwargs'] = self.int_clip_symbolic_kwargs(
                    module.is_narrow_range, module.is_signed, module.bit_width())
            else:
                self.symbolic_kwargs['clip_symbolic_kwargs'] = self.float_clip_symbolic_kwargs(
                    module.is_narrow_range, 
                    module.is_signed, 
                    module.bit_width(),
                    # preserve broadcastable shape if per-channel, 0-dim otherwise
                    to_0dim_if_scalar(module.scale()), 
                    to_0dim_if_scalar(module.zero_point()))
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
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(scale, zero_point, bit_width)

        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        if clip_symbolic_kwargs is not None and self.clip_over_integers:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        if clip_symbolic_kwargs is not None and not self.clip_over_integers:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class QCDQWeightQuantProxyHandlerMixin(QCDQQuantProxyHandlerMixin):
    handled_layer = WeightQuantProxyFromInjector


class QCDQDecoupledWeightQuantProxyHandlerMixin(QCDQWeightQuantProxyHandlerMixin):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def quantize_symbolic_kwargs(cls, module):
        flat_scale = to_0dim_if_scalar(module.pre_scale().flatten())
        zp = to_0dim_if_scalar(module.pre_zero_point().flatten()).expand_as(flat_scale)
        return {
            'scale': flat_scale,
            'zero_point': cls.zero_point_with_dtype(module.is_signed, zp),
            'dtype': cls.int8_dtype() if module.is_signed else cls.uint8_dtype(),
            'axis': cls.quant_axis(module.pre_scale())}

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        pre_scale = quantize_symbolic_kwargs['scale']
        pre_zero_point = quantize_symbolic_kwargs['zero_point']
        return out, pre_scale, pre_zero_point, scale, zero_point, bit_width


class QCDQActQuantProxyHandlerMixin(QCDQQuantProxyHandlerMixin):
    handled_layer = ActQuantProxyFromInjector


class QCDQBiasQuantProxyHandlerMixin(DQMixin, QuantAxisMixin):
    handled_layer = BiasQuantProxyFromInjector

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
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(scale, zero_point, bit_width)

        dtype = torch.int32 if int(bit_width.item()) > 8 else torch.int8
        quant_axis = self.quant_axis(scale)
        if self.flatten_dequantize_params:
            scale = to_0dim_if_scalar(scale.flatten())
            zero_point = to_0dim_if_scalar(zero_point.flatten()).expand_as(scale).to(dtype)
        y = self.dequantize_fn(
            int_bias.to(dtype), scale, zero_point, quant_axis)
        return y, scale, zero_point, bit_width


class QCDQTruncQuantProxyHandlerMixin(QCDQQuantProxyHandlerMixin):
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
        dtype = self.int8_dtype() if signed else self.uint8_dtype()
        flat_scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(flat_scale)
        x = self.quantize_fn(x, flat_scale, zp, dtype, self.quant_axis(scale))
        clip_symbolic_kwargs = self.clip_symbolic_kwargs(signed, False, output_bit_width)
        if clip_symbolic_kwargs is not None and self.clip_over_integers:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, flat_scale, zp, self.quant_axis(scale))
        if clip_symbolic_kwargs is not None and not self.clip_over_integers:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        return x, scale, zero_point, output_bit_width