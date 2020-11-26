from typing import Union
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from brevitas.nn import QuantReLU, QuantIdentity, QuantHardTanh, QuantTanh, QuantSigmoid
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from ..function import QuantizeLinearFunction, DequantizeLinearFunction
from .base import StdONNXQuantLayerHandler


class StdONNXQuantNLALHandler(StdONNXQuantLayerHandler, ABC):

    @classmethod
    def validate(cls, module: QuantNLAL):
        input_bit_width = module.quant_input_bit_width()
        act_bit_width = module.quant_act_bit_width()
        if input_bit_width is not None:
            cls.validate_8b_bit_width(input_bit_width)
        if act_bit_width is not None:
            cls.validate_8b_bit_width(act_bit_width)

    def prepare_for_export(self, module: QuantNLAL):
        self.validate(module)
        if not module.return_quant_tensor:
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
        else:
            output_dequant_symbolic_kwargs = None
        output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)
        input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        input_quant_symbolic_kwargs = self.input_quant_symbolic_kwargs(module)
        if input_quant_symbolic_kwargs is not None:
            input_redequant_symbolic_kwargs = {
                'input_scale': input_quant_symbolic_kwargs['output_scale'],
                'input_zero_point': input_quant_symbolic_kwargs['output_zero_point'],
                'axis': input_quant_symbolic_kwargs['axis']}
        else:
            input_redequant_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'input_quant_symbolic_kwargs': input_quant_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'input_redequant_symbolic_kwargs': input_redequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        input_quant_symbolic_kwargs = self.symbolic_kwargs['input_quant_symbolic_kwargs']
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        input_redequant_symbolic_kwargs = self.symbolic_kwargs['input_redequant_symbolic_kwargs']
        if input_dequant_symbolic_kwargs is not None:
            inp = DequantizeLinearFunction.apply(inp, *input_dequant_symbolic_kwargs.values())
        if input_quant_symbolic_kwargs is not None:
            inp = QuantizeLinearFunction.apply(inp, *input_quant_symbolic_kwargs.values())
            inp = DequantizeLinearFunction.apply(inp, *input_redequant_symbolic_kwargs.values())
        return inp

    def output_symbolic_execution(self, out: Tensor):
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        out = QuantizeLinearFunction.apply(out, *output_quant_symbolic_kwargs.values())
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFunction.apply(out, *output_dequant_symbolic_kwargs.values())
        return out


class StdONNXQuantReLUHandler(StdONNXQuantNLALHandler):
    handled_layer = QuantReLU

    def op_symbolic_execution(self, inp: Tensor):
        return torch.relu(inp)


class StdONNXQuantTanhHandler(StdONNXQuantNLALHandler):
    handled_layer = QuantTanh

    def op_symbolic_execution(self, inp: Tensor):
        return torch.tanh(inp)


class StdONNXQuantSigmoidHandler(StdONNXQuantNLALHandler):
    handled_layer = QuantSigmoid

    def op_symbolic_execution(self, inp: Tensor):
        return torch.sigmoid(inp)


class StdONNXQuantIdentityHandler(StdONNXQuantLayerHandler):
    handled_layer = QuantIdentity

    @classmethod
    def validate(cls, module: QuantNLAL):
        assert not module.is_input_quant_enabled  # not supported
        if module.is_act_quant_enabled:
            cls.validate_8b_bit_width(module.quant_act_bit_width())
        else:
            assert module._cached_out is not None

    def prepare_for_export(self, module: QuantNLAL):
        self.validate(module)
        input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)
        if not module.return_quant_tensor:
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
        else:
            output_dequant_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        return inp

    def op_symbolic_execution(self, inp: Tensor):
        return inp

    def output_symbolic_execution(self, out: Tensor):
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        if input_dequant_symbolic_kwargs:
            out = DequantizeLinearFunction.apply(out, *input_dequant_symbolic_kwargs.values())
        out = QuantizeLinearFunction.apply(out, *output_quant_symbolic_kwargs.values())
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFunction.apply(out, *output_dequant_symbolic_kwargs.values())
        return out


class StdONNXQuantHardTanhHandler(StdONNXQuantIdentityHandler):
    handled_layer = QuantHardTanh