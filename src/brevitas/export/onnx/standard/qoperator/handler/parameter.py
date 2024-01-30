from abc import ABC
from abc import abstractmethod
from typing import Union

import torch
from torch import Tensor

from brevitas.export.common import to_0dim_if_scalar
from brevitas.export.onnx.handler import Kernel1dApplHandlerMixin
from brevitas.export.onnx.handler import Kernel2dApplHandlerMixin
from brevitas.export.onnx.standard.function import DequantizeLinearFn
from brevitas.export.onnx.standard.function import IntClipFn
from brevitas.export.onnx.standard.function import QuantizeLinearFn
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

from ..function import QLinearConvFn
from ..function import QLinearMatMulFn
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
        assert module.is_weight_quant_enabled, 'Weight quant required'
        assert module.is_output_quant_enabled, 'Output quant required'
        # Handling narrow_range is across the network is difficult do to the fact that
        # it's not part of QuantTensor, and so it can't be cached
        assert not module.is_quant_output_narrow_range, 'Narrow output quant not supported'
        if module.is_input_quant_enabled:
            assert not module.is_quant_input_narrow_range, 'Narrow output quant not supported'
        cls.validate_8b_bit_width(module.quant_weight_bit_width(), le_then=True)
        cls.validate_8b_bit_width(module.quant_input_bit_width(), le_then=True)
        cls.validate_8b_bit_width(module.quant_output_bit_width(), le_then=True)
        if module.bias is not None and requires_quant_bias:
            assert module.is_bias_quant_enabled
            assert module.is_quant_bias_signed
            cls.validate_32b_bit_width(module.quant_bias_bit_width(), le_then=True)

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
            input_clip_symbolic_kwargs = self.input_clip_symbolic_kwargs(module)
        else:
            input_dequant_symbolic_kwargs = None
            input_clip_symbolic_kwargs = None
        output_clip_symbolic_kwargs = self.output_clip_symbolic_kwargs(module)

        self.symbolic_kwargs = {
            'op_symbolic_kwargs': op_symbolic_kwargs,
            'input_quant_symbolic_kwargs': input_quant_symbolic_kwargs,
            'input_clip_symbolic_kwargs': input_clip_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs,
            'output_clip_symbolic_kwargs': output_clip_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        input_quant_symbolic_kwargs = self.symbolic_kwargs['input_quant_symbolic_kwargs']
        input_clip_symbolic_kwargs = self.symbolic_kwargs['input_clip_symbolic_kwargs']
        if input_dequant_symbolic_kwargs is not None:
            assert input_quant_symbolic_kwargs is not None
            inp = DequantizeLinearFn.apply(inp, *input_dequant_symbolic_kwargs.values())
        if input_quant_symbolic_kwargs is not None:
            inp = QuantizeLinearFn.apply(inp, *input_quant_symbolic_kwargs.values())
            if input_clip_symbolic_kwargs is not None:
                inp = IntClipFn.apply(inp, *input_clip_symbolic_kwargs.values())
        return inp

    def output_symbolic_execution(self, out: Tensor):
        output_clip_symbolic_kwargs = self.symbolic_kwargs['output_clip_symbolic_kwargs']
        if output_clip_symbolic_kwargs is not None:
            out = IntClipFn.apply(out, *output_clip_symbolic_kwargs.values())
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFn.apply(out, *output_dequant_symbolic_kwargs.values())
        return out


class StdQOpONNXQuantConvNdHandler(StdQOpONNXQuantWBIOLHandler, ABC):

    def op_symbolic_kwargs(self, module: Union[QuantConv1d, QuantConv2d]):
        conv_symbolic_kwargs = {
            'input_scale': module.quant_input_scale(),
            'input_zero_point': self.quant_input_zero_point(module),
            'int_weight': self.int_weight(module),
            'weight_scale': to_0dim_if_scalar(module.quant_weight_scale().flatten()),
            'weight_zero_point': to_0dim_if_scalar(self.quant_weight_zero_point(module).flatten()),
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

    def op_symbolic_execution(self, inp: Tensor):
        conv_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        out = QLinearConvFn.apply(inp, *conv_symbolic_kwargs.values())
        return out


class StdQOpONNXQuantConv2dHandler(StdQOpONNXQuantConvNdHandler, Kernel2dApplHandlerMixin):
    handled_layer = QuantConv2d


class StdQOpONNXQuantConv1dHandler(StdQOpONNXQuantConvNdHandler, Kernel1dApplHandlerMixin):
    handled_layer = QuantConv1d


class StdQOpONNXQuantLinearHandler(StdQOpONNXQuantWBIOLHandler):
    handled_layer = QuantLinear

    # Convert linear to conv1d to handle bias
    def op_symbolic_kwargs(self, module: QuantLinear):
        conv_symbolic_kwargs = {
            'input_scale': module.quant_input_scale(),
            'input_zero_point': self.quant_input_zero_point(module),
            'int_weight': self.int_weight(module).view(module.out_features, module.in_features, 1),
            'weight_scale': to_0dim_if_scalar(module.quant_weight_scale().flatten()),
            'weight_zero_point': to_0dim_if_scalar(self.quant_weight_zero_point(module).flatten()),
            'output_scale': module.quant_output_scale(),
            'output_zero_point': self.quant_output_zero_point(module),
            'output_dtype': self.torch_8b_dtype(module.is_quant_output_signed),
            'int_bias': self.int_bias(module),
            'out_shape': self.quant_output_shape(module) + (1,),
            'kernel_size': [1],
            'padding': [0, 0],
            'stride': [1],
            'groups': 1,
            'dilation': [1]}
        return conv_symbolic_kwargs

    def op_symbolic_execution(self, inp):
        linear_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        inp = inp.view(inp.size(0), -1, 1)
        out = QLinearConvFn.apply(inp, *linear_symbolic_kwargs.values())
        out = out.view(out.size(0), -1)
        return out
