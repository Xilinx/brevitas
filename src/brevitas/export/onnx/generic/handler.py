from abc import ABC
from copy import copy

from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector

from .function import BrevitasQuantFn, BrevitasBinaryQuantFn, BrevitasTruncFn


class BrevitasQuantProxyHandler(ONNXBaseHandler, ABC):

    def validate(self, module):
        if module.bit_width() == 1:
            assert module.zero_point() == 0, "Zero-point not supported for binary quant."

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width(),
                'narrow_range': module.is_narrow_range,
                'signed': module.is_signed,
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(self, x: Tensor):
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        if bit_width == 1:
            x = BrevitasBinaryQuantFn.apply(x, *self.symbolic_kwargs.values())
        else:
            x = BrevitasQuantFn.apply(x, *self.symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class BrevitasWeightQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.quant_weights = None

    def reset(self):
        super().reset()
        self.quant_weights = None

    def prepare_for_export(self, module: WeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        quant_weights = {
            tm.weight.data_ptr(): tm.quant_weight().value for tm in module.tracked_module_list}
        self.quant_weights = quant_weights
        # override rounding mode since quantization has been pre-applied
        self.symbolic_kwargs['rounding_mode'] = 'ROUND'

    def symbolic_execution(self, x: Tensor):
        quant_weight = self.quant_weights[x.data_ptr()]
        return super().symbolic_execution(quant_weight)


class BrevitasDecoupledWeightQuantProxyHandler(BrevitasWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.extra_kwargs = {}

    def reset(self):
        super().reset()
        self.extra_kwargs = {}

    def prepare_for_export(self, module: DecoupledWeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        self.extra_kwargs['pre_scale'] = module.pre_scale()
        self.extra_kwargs['pre_zero_point'] = module.pre_zero_point()

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        pre_scale = self.extra_kwargs['pre_scale']
        pre_zero_point = self.extra_kwargs['pre_zero_point']
        return out, pre_scale, pre_zero_point, scale, zero_point, bit_width


class BrevitasActQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector


class BrevitasBiasQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        # avoid in place pop in case the proxy is shared
        symbolic_kwargs = copy(self.symbolic_kwargs)
        scale = symbolic_kwargs.pop('scale')
        bit_width = symbolic_kwargs.pop('bit_width')
        zero_point = symbolic_kwargs.pop('zero_point')
        if scale is None:
            assert input_scale is not None, 'Input scale required for bias export'
            scale = input_scale
        if bit_width is None:
            assert input_bit_width is not None, 'Input bit_width required for bias export'
            bit_width = input_bit_width
        y = BrevitasQuantFn.apply(
            x, scale, zero_point, bit_width, *symbolic_kwargs.values())
        return y, scale, zero_point, bit_width


class BrevitasTruncQuantProxyHandler(ONNXBaseHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        self.symbolic_kwargs = {
                'output_bit_width': module.bit_width(),
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor):
        y = BrevitasTruncFn.apply(
            x, scale, zero_point, input_bit_width, *self.symbolic_kwargs.values())
        return y, scale, zero_point, self.symbolic_kwargs['output_bit_width']