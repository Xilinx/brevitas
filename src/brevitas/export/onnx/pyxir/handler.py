from abc import ABC
from typing import Union
import math
import inspect

from torch import Tensor
import torch.nn.functional as F

from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn import QuantConv2d, QuantReLU, QuantEltwiseAdd, QuantMaxPool2d, QuantLinear
from brevitas.nn import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from brevitas.export.onnx.handler import ONNXBaseHandler, Kernel2dApplHandler
from .function import DPUQuantReLUPlaceholderFunction, DPUQuantEltwiseAddPlaceholderFunction
from .function import DPUQuantAvgPoolPlaceholderFunction, DPUQuantLinearPlaceholderFunction


class DPUQuantLayerHandler(ONNXBaseHandler, ABC):

    @staticmethod
    def neg_scalar_exponent_from_scale(scale: Tensor):
        if scale is None:
            raise RuntimeError("Scale cannot be None")
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError("Only per-tensor scaling is currently supported")
        scale = scale.item()
        neg_exponent = - math.log2(scale)
        if not neg_exponent.is_integer():
            raise RuntimeError("Only power-of-two scale factors are supported")
        neg_exponent = int(neg_exponent)
        return neg_exponent

    @staticmethod
    def validate_8b_bit_width(bit_width: Tensor):
        if bit_width is None:
            raise RuntimeError("Bit width cannot be None")
        bit_width = int(bit_width.item())
        return bit_width

    @staticmethod
    def quant_input_scale(module: QuantLayerMixin):
        scale = module.quant_input_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_output_scale(module: QuantLayerMixin):
        scale = module.quant_output_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_input_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)


    @staticmethod
    def quant_output_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_output_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_output_shape(module: QuantLayerMixin):
        cached_out = module._cached_out  # TODO add shape property to the module
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required")
        return cached_out.shape

    def prepare_from_cached_io(self, cached_io):
        cached_inp, cached_out = cached_io
        self.symbolic_kwargs = {
            'output_shape': cached_out.shape,
            'input_bit_width': self.validate_8b_bit_width(cached_inp.bit_width),
            'input_scale': self.neg_scalar_exponent_from_scale(cached_inp.scale),
            'output_bit_width': self.validate_8b_bit_width(cached_out.bit_width),
            'output_scale': self.neg_scalar_exponent_from_scale(cached_out.scale)
        }


class DPUQuantReLUHandler(DPUQuantLayerHandler):
    handled_layer = QuantReLU

    def prepare_for_export(self, module: QuantReLU):
        self.symbolic_kwargs = {
            'output_shape': self.quant_output_shape(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = DPUQuantReLUPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret

    def cached_symbolic_execution(self, inp: Tensor, *args, **kwargs):
        kwargs.update(self.symbolic_kwargs)
        if 'inplace' in kwargs:
            del kwargs['inplace']
        return DPUQuantReLUPlaceholderFunction.apply(inp, *args, *kwargs.values())


class DPUQuantEltwiseAddHandler(DPUQuantLayerHandler):
    handled_layer = QuantEltwiseAdd

    def prepare_for_export(self, module: QuantEltwiseAdd):
        self.symbolic_kwargs = {
            'other_bit_width': self.quant_input_bit_width(module),
            'other_scale': self.quant_input_scale(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module)}

    def symbolic_execution(self, inp: Tensor, other: Tensor):
        ret = DPUQuantEltwiseAddPlaceholderFunction.apply(
            inp, other, *self.symbolic_kwargs.values())
        return ret


class DPUQuantMaxPool2dHandler(DPUQuantLayerHandler, Kernel2dApplHandler, ABC):
    handled_layer = QuantMaxPool2d

    @staticmethod
    def _solve_max_pool2d_kwargs(inp, args, kwargs):
        signature = inspect.signature(F._max_pool2d)
        ba = signature.bind(inp, *args, **kwargs)
        ba.apply_defaults()
        if 'return_indices' in ba.arguments:
            assert not ba.arguments['return_indices']
            del ba.arguments['return_indices']
        return ba.arguments

    def prepare_for_export(self, module: QuantMaxPool2d):
        self.symbolic_kwargs = {
            'kernel_shape': self.kernel_shape(module),
            'pads': self.padding(module),
            'strides': self.stride(module),
            'ceil_mode': module.ceil_mode,
            'dilations': self.dilation(module),
            'output_shape': self.quant_output_shape(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module)}


class DPUQuantAvgPool2dHandler(DPUQuantLayerHandler, Kernel2dApplHandler):
    handled_layer = (QuantAvgPool2d, QuantAdaptiveAvgPool2d)

    def prepare_for_export(self, module: Union[QuantAvgPool2d, QuantAdaptiveAvgPool2d]):
        self.symbolic_kwargs = {
            'kernel_shape': self.kernel_shape(module),  # from caching
            'strides': self.stride(module),  # from caching
            'pads': self.padding(module),  # supposed to be always 0
            'output_shape': self.quant_output_shape(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = DPUQuantAvgPoolPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret
        

class DPUQuantWeightBiasHandler(ABC):
    
    @staticmethod
    def int_weight(module: QuantConv2d):
        return module.int_weight(float_datatype=False).detach()

    @staticmethod
    def quant_weight_bit_width(module: QuantWBIOL):
        bit_width = module.quant_weight_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_weight_scale(module: QuantWBIOL):
        quant_weight_scale = module.quant_weight_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(quant_weight_scale)

    @staticmethod
    def int_bias(module: QuantWBIOL):
        if module.bias is not None:
            return module.int_bias(float_datatype=False).detach()
        else:
            return None

    @staticmethod
    def quant_bias_bit_width(module: QuantWBIOL):
        if module.bias is not None:
            bit_width = module.quant_bias_bit_width()
            return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)
        else:
            return None

    @staticmethod
    def quant_bias_scale(module: QuantWBIOL):
        if module.bias is not None:
            scale = module.quant_bias_scale()
            return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)
        else:
            return None


class DPUQuantLinearHandler(DPUQuantLayerHandler, DPUQuantWeightBiasHandler):
    handled_layer = QuantLinear

    def prepare_for_export(self, module: QuantAdaptiveAvgPool2d):
        self.symbolic_kwargs = {
            'int_weight': self.int_weight(module),
            'int_bias': self.int_bias(module),
            'output_shape': self.quant_output_shape(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module),
            'weight_bit_width': self.quant_weight_bit_width(module),
            'weight_scale': self.quant_weight_scale(module),
            'bias_bit_width': self.quant_bias_bit_width(module),
            'bias_scale': self.quant_bias_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = DPUQuantLinearPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUQuantConv2dHandler(
    DPUQuantLayerHandler,
    DPUQuantWeightBiasHandler,
    Kernel2dApplHandler, 
    ABC):
    handled_layer = QuantConv2d

    def prepare_for_export(self, module):
        self.symbolic_kwargs = {
            'int_weight': self.int_weight(module),
            'int_bias': self.int_bias(module),
            'out_shape': self.quant_output_shape(module),
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module),
            'weight_bit_width': self.quant_weight_bit_width(module),
            'weight_scale': self.quant_weight_scale(module),
            'bias_bit_width': self.quant_bias_bit_width(module),
            'bias_scale': self.quant_bias_scale(module),
            'kernel_size': list(module.kernel_size),
            'padding': self.padding(module),
            'stride': self.stride(module),
            'groups': module.groups,
            'dilation': self.dilation(module)}



