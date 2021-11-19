from typing import Union
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn import QuantConv2d, QuantConv1d, QuantLinear
from brevitas.export.onnx.handler import Kernel2dApplHandlerMixin, Kernel1dApplHandlerMixin
from brevitas.export.onnx.standard.function import QuantizeLinearFn, DequantizeLinearFn
from ..function import QLinearConvFn, QLinearMatMulFn
from .base import StdQOpONNXQuantLayerHandler


class StdQOpONNXQuantWBIOLHandler(StdQOpONNXQuantLayerHandler, ABC):

    @staticmethod
    def int_weight(module: QuantWBIOL):
        int_weight = module.int_weight(float_datatype=False).detach()
        if module.is_quant_weight_signed:
            return int_weight.type(torch.int8)
        else:
            return int_weight.type(torch.uint8)

    @staticmethod
    def int_bias(module: QuantWBIOL):
        if module.bias is not None:
            int_bias = module.int_bias(float_datatype=False).detach()
            return int_bias.type(torch.int32)
        else:
            return None

    @classmethod
    def validate(cls, module: QuantWBIOL, requires_quant_bias=True):
        assert module.is_weight_quant_enabled
        assert module.is_output_quant_enabled
        cls.validate_8b_bit_width(module.quant_weight_bit_width(), le_then=True)
        cls.validate_8b_bit_width(module.quant_input_bit_width())
        cls.validate_8b_bit_width(module.quant_output_bit_width())
        if module.bias is not None and requires_quant_bias:
            assert module.is_bias_quant_enabled
            assert module.is_quant_bias_signed
            cls.validate_32b_bit_width(module.quant_bias_bit_width(), le_then=True)

    def input_symbolic_execution(self, inp: Tensor):
        input_quant_symbolic_kwargs = self.symbolic_kwargs['input_quant_symbolic_kwargs']
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        if input_dequant_symbolic_kwargs is not None:
            assert input_quant_symbolic_kwargs is not None
            inp = DequantizeLinearFn.apply(inp, *input_dequant_symbolic_kwargs.values())
        if input_quant_symbolic_kwargs is not None:
            inp = QuantizeLinearFn.apply(inp, *input_quant_symbolic_kwargs.values())
        return inp


class StdQOpONNXQuantLinearHandler(StdQOpONNXQuantWBIOLHandler):
    handled_layer = QuantLinear

    @classmethod
    def op_symbolic_kwargs(cls, module: QuantLinear):
        linear_symbolic_kwargs = {
            'input_scale': module.quant_input_scale(),
            'input_zero_point': cls.quant_input_zero_point(module),
            'int_weight': cls.int_weight(module).t(),
            'weight_scale': module.quant_weight_scale(),
            'weight_zero_point': cls.quant_weight_zero_point(module),
            'output_scale': module.quant_output_scale(),
            'output_zero_point': cls.quant_output_zero_point(module),
            'output_dtype': cls.torch_8b_dtype(module.is_quant_output_signed),
            'out_shape': cls.quant_output_shape(module)}
        return linear_symbolic_kwargs

    def prepare_for_export(self, module: QuantLinear):
        self.validate(module, requires_quant_bias=False)

        op_symbolic_kwargs = self.op_symbolic_kwargs(module)
        input_quant_symbolic_kwargs = self.input_quant_symbolic_kwargs(module)
        if input_quant_symbolic_kwargs is not None:
            input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        else:
            input_dequant_symbolic_kwargs = None

        if not module.return_quant_tensor:
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
            output_quant_symbolic_kwargs = None
        elif module.return_quant_tensor and not module.bias is not None:
            output_dequant_symbolic_kwargs = None
            output_quant_symbolic_kwargs = None
        else:
            assert module.return_quant_tensor and module.bias is not None
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
            output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)

        self.symbolic_kwargs = {
            'bias': module.bias,
            'op_symbolic_kwargs': op_symbolic_kwargs,
            'input_quant_symbolic_kwargs': input_quant_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs}

    def op_symbolic_execution(self, inp):
        linear_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        out = QLinearMatMulFn.apply(inp, *linear_symbolic_kwargs.values())
        return out

    def output_symbolic_execution(self, out: Tensor):
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        bias = self.symbolic_kwargs['bias']
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFn.apply(out, *output_dequant_symbolic_kwargs.values())
        if bias is not None:
            out = out.add(bias)
        if output_quant_symbolic_kwargs is not None:
            out = QuantizeLinearFn.apply(out, *output_quant_symbolic_kwargs.values())
        return out


class StdQOpONNXQuantConvNdHandler(StdQOpONNXQuantWBIOLHandler, ABC):

    def op_symbolic_kwargs(self, module: Union[QuantConv1d, QuantConv2d]):
        conv_symbolic_kwargs = {
            'input_scale': module.quant_input_scale(),
            'input_zero_point': self.quant_input_zero_point(module),
            'int_weight': self.int_weight(module),
            'weight_scale': module.quant_weight_scale(),
            'weight_zero_point': self.quant_weight_zero_point(module),
            'output_scale': module.quant_output_scale(),
            'output_zero_point': self.quant_output_zero_point(module),
            'output_dtype': self.torch_8b_dtype(module.is_quant_output_signed),
            'int_bias': self.int_bias(module),
            'out_shape': self.quant_output_shape(module),
            'kernel_size': list(module.kernel_size),
            'padding': self.padding(module),
            'stride': self.stride(module),
            'groups': module.groups,
            'dilation': self.dilation(module)}
        return conv_symbolic_kwargs

    def prepare_for_export(self, module: Union[QuantConv1d, QuantConv2d]):
        self.validate(module)

        op_symbolic_kwargs = self.op_symbolic_kwargs(module)
        if not module.return_quant_tensor:
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
        else:
            output_dequant_symbolic_kwargs = None
        input_quant_symbolic_kwargs = self.input_quant_symbolic_kwargs(module)
        if input_quant_symbolic_kwargs is not None:
            input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        else:
            input_dequant_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'op_symbolic_kwargs': op_symbolic_kwargs,
            'input_quant_symbolic_kwargs': input_quant_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs}

    def op_symbolic_execution(self, inp: Tensor):
        conv_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        out = QLinearConvFn.apply(inp, *conv_symbolic_kwargs.values())
        return out

    def output_symbolic_execution(self, out: Tensor):
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFn.apply(out, *output_dequant_symbolic_kwargs.values())
        return out


class StdQOpONNXQuantConv2dHandler(StdQOpONNXQuantConvNdHandler, Kernel2dApplHandlerMixin):
    handled_layer = QuantConv2d


class StdQOpONNXQuantConv1dHandler(StdQOpONNXQuantConvNdHandler, Kernel1dApplHandlerMixin):
    handled_layer = QuantConv1d
