from typing import Union
from abc import ABC, abstractmethod

import torch
from torch import Tensor


from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.common.handler import Validate8BitHandler, TypedZeroPointHandler
from ..function import QuantizeLinearFunction, DequantizeLinearFunction


class StdONNXQuantLayerHandler(Validate8BitHandler, TypedZeroPointHandler, ONNXBaseHandler, ABC):

    @abstractmethod
    def op_symbolic_execution(self, inp: Tensor):
        pass

    @abstractmethod
    def input_symbolic_execution(self, inp: Tensor):
        pass

    @abstractmethod
    def output_symbolic_execution(self, out: Tensor):
        pass

    @classmethod
    def op_symbolic_kwargs(cls, module):
        raise NotImplementedError  # optional method

    @classmethod
    def quant_output_shape(cls, module):
        cached_out = module._cached_out
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required to export QuantConv2d")
        return cached_out.shape

    @classmethod
    def quant_axis(cls, scale):
        for i in scale.shape:
            if i != 1:
                return i
        return None

    @classmethod
    def output_quant_symbolic_kwargs(cls, module):
        return {
            'output_scale': module.quant_output_scale(),
            'output_zero_point': cls.quant_output_zero_point(module),
            'axis': cls.quant_axis(module.quant_output_scale())}

    @classmethod
    def output_dequant_symbolic_kwargs(cls, module):
        return {
            'input_scale': module.quant_output_scale(),
            'input_zero_point': cls.quant_output_zero_point(module),
            'axis': cls.quant_axis(module.quant_output_scale())}

    @classmethod
    def input_quant_symbolic_kwargs(cls, module):
        if module.is_input_quant_enabled:
            return {
                'output_scale': module.quant_input_scale(),
                'output_zero_point': cls.quant_input_zero_point(module),
                'axis': cls.quant_axis(module.quant_input_scale())}
        else:
            return None

    @classmethod
    def input_dequant_symbolic_kwargs(cls, module):
        if module._cached_inp.scale is not None:
            assert module._cached_inp.bit_width == 8
            return {
                'input_scale': module._cached_inp.scale,
                'input_zero_point': cls.zero_point_with_dtype(
                    module._cached_inp.signed, module._cached_inp.zero_point),
                'axis': cls.quant_axis(module._cached_inp.scale)}
        else:
            return None

    def symbolic_execution(self, inp: Tensor):
        inp = self.input_symbolic_execution(inp)
        out = self.op_symbolic_execution(inp)
        ret = self.output_symbolic_execution(out)
        return ret


class StdONNXQuantWrapperHandler(StdONNXQuantLayerHandler, ABC):

    @classmethod
    def validate(cls, module):
        cls.validate_8b_bit_width(module.quant_input_bit_width())
        cls.validate_8b_bit_width(module.quant_output_bit_width())

    def prepare_for_export(self, module):
        self.validate(module)
        op_symbolic_kwargs = self.op_symbolic_kwargs(module)
        input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        if module.return_quant_tensor:
            output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)
        else:
            output_quant_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'op_symbolic_kwargs': op_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        inp = DequantizeLinearFunction.apply(inp, *input_dequant_symbolic_kwargs.values())
        return inp

    def output_symbolic_execution(self, out: Tensor):
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        if output_quant_symbolic_kwargs is not None:
            out = QuantizeLinearFunction.apply(out, *output_quant_symbolic_kwargs.values())
        return out

