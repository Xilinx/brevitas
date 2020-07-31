from abc import ABC
from typing import Tuple
import math

import torch
from torch import Tensor

from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.nn import QuantConv2d, QuantReLU
from brevitas.onnx.base import BaseHandler
from .function import QuantizedConv2dPlaceholderFunction, QuantizedReLUPlaceholderFunction


class DPUv1QuantLayerHandler(BaseHandler, ABC):

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
        if bit_width != 8:
            raise RuntimeError("Only 8b bit width supported")
        return bit_width

    @staticmethod
    def quant_input_scale(module: QuantLayerMixin):
        scale = module.quant_input_scale()
        return DPUv1QuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_output_scale(module: QuantLayerMixin):
        scale = module.quant_output_scale()
        return DPUv1QuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_input_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return DPUv1QuantLayerHandler.validate_8b_bit_width(bit_width)


    @staticmethod
    def quant_output_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_output_bit_width()
        return DPUv1QuantLayerHandler.validate_8b_bit_width(bit_width)


class DPUv1QuantReLUHandler(DPUv1QuantLayerHandler):
    handled_layer = QuantReLU

    def prepare_for_symbolic_execution(self, module):
        self.symbolic_kwargs = {
            'input_bit_width': self.quant_input_bit_width(module),
            'input_scale': self.quant_input_scale(module),
            'output_bit_width': self.quant_output_bit_width(module),
            'output_scale': self.quant_output_scale(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedReLUPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class DPUv1QuantConv2dHandler(DPUv1QuantLayerHandler):
    handled_layer = QuantConv2d

    @staticmethod
    def quant_output_shape(module: QuantConv2d):
        cached_out = module._cached_out  # TODO add shape property to the module
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required to export QuantConv2d")
        return cached_out.shape

    @staticmethod
    def int_weight(module: QuantConv2d):
        return module.int_weight(float_datatype=False).detach()

    @staticmethod
    def quant_weight_bit_width(module: QuantConv2d):
        bit_width = module.quant_weight_bit_width()
        return DPUv1QuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_weight_scale(module: QuantConv2d):
        quant_weight_scale = module.quant_weight_scale()
        return DPUv1QuantConv2dHandler.neg_scalar_exponent_from_scale(quant_weight_scale)

    @staticmethod
    def int_bias(module: QuantConv2d):
        return module.int_bias(float_datatype=False).detach()

    @staticmethod
    def quant_bias_bit_width(module: QuantConv2d):
        bit_width = module.quant_bias_bit_width()
        return DPUv1QuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_bias_scale(module: QuantConv2d):
        scale = module.quant_bias_scale()
        return DPUv1QuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 4
        else:
            padding = list(module.padding) + list(module.padding)
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride] * 2
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module: QuantConv2d):
        if not module.dilation == (1, 1):
            raise RuntimeError("Dilation is not supported")
        return list(module.dilation)

    def prepare_for_symbolic_execution(self, module):
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

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedConv2dPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret



