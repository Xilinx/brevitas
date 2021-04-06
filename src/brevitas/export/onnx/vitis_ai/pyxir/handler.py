from abc import ABC
from typing import Union
import inspect

from torch import Tensor
import torch.nn.functional as F

from brevitas.nn import QuantConv2d, QuantReLU, QuantEltwiseAdd, QuantMaxPool2d, QuantLinear
from brevitas.nn import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from brevitas.export.onnx.handler import Kernel2dApplHandler
from brevitas.export.onnx.vitis_ai.handler import DPUQuantLayerHandler, DPUQuantWeightBiasHandler
from .function import DPUQuantReLUPlaceholderFunction, DPUQuantEltwiseAddPlaceholderFunction
from .function import DPUQuantAvgPoolPlaceholderFunction, DPUQuantLinearPlaceholderFunction


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



