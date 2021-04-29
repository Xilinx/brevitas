from abc import ABC

from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector

from .function import QuantPlaceholderFunction, TruncPlaceholderFunction


class StaticQuantProxyHandler(ONNXBaseHandler, ABC):

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.symbolic_kwargs = {
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width(),
                'narrow_range': module.is_narrow_range,
                'signed': module.is_signed}

    def symbolic_execution(self, x: Tensor):
        x = QuantPlaceholderFunction.apply(x, *self.symbolic_kwargs.values())
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        return x, scale, zero_point, bit_width


class WeightQuantProxyHandler(StaticQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector


class ActQuantProxyHandler(StaticQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector


class BiasQuantProxyHandler(StaticQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        scale = self.symbolic_kwargs.pop('scale')
        bit_width = self.symbolic_kwargs.pop('bit_width')
        zero_point = self.symbolic_kwargs.pop('zero_point')
        if scale is None:
            assert input_scale is not None, 'Input scale required for bias export'
            scale = input_scale
        if bit_width is None:
            assert input_bit_width is not None, 'Input bit_width required for bias export'
            bit_width = input_bit_width
        x = QuantPlaceholderFunction.apply(
            x, scale, zero_point, bit_width, *self.symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class TruncQuantProxyHandler(ONNXBaseHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        self.symbolic_kwargs = {
                'bit_width': module.bit_width()}

    def symbolic_execution(self, x: Tensor, scale: Tensor, zero_point: Tensor, bit_width: Tensor):
        x = TruncPlaceholderFunction.apply(
            x, scale, zero_point, *self.symbolic_kwargs.values())
        return x, scale, zero_point, self.symbolic_kwargs['bit_width']