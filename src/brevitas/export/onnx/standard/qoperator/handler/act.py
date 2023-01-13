from abc import ABC

import torch
from torch import Tensor

from brevitas.export.onnx.standard.function import DequantizeLinearFn
from brevitas.export.onnx.standard.function import IntClipFn
from brevitas.export.onnx.standard.function import QuantizeLinearFn
from brevitas.nn import QuantHardTanh
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantReLU
from brevitas.nn import QuantSigmoid
from brevitas.nn import QuantTanh
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

from .base import StdQOpONNXQuantLayerHandler


class StdQOpONNXQuantNLALHandler(StdQOpONNXQuantLayerHandler, ABC):

    @classmethod
    def validate(cls, module: QuantNLAL):
        if cls.input_quant_supported and module.is_input_quant_enabled:
            assert not module.is_quant_input_narrow_range, "Narrow range quant not supported."
        elif not cls.input_quant_supported and module.is_input_quant_enabled:
            raise RuntimeError("Input quant not supported.")
        if module.is_act_quant_enabled:
            assert not module.is_quant_act_narrow_range, "Narrow range quant not supported."
        input_bit_width = module.quant_input_bit_width()
        act_bit_width = module.quant_act_bit_width()
        if input_bit_width is not None:
            cls.validate_8b_bit_width(input_bit_width, le_then=True)
        if act_bit_width is not None:
            cls.validate_8b_bit_width(act_bit_width, le_then=True)

    def prepare_for_export(self, module: QuantNLAL):
        self.validate(module)
        if not module.return_quant_tensor:
            output_dequant_symbolic_kwargs = self.output_dequant_symbolic_kwargs(module)
        else:
            output_dequant_symbolic_kwargs = None
        output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)
        output_clip_symbolic_kwargs = self.output_clip_symbolic_kwargs(module)
        input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        input_quant_symbolic_kwargs = self.input_quant_symbolic_kwargs(module)
        input_clip_symbolic_kwargs = self.input_clip_symbolic_kwargs(module)
        if input_quant_symbolic_kwargs is not None:
            input_redequant_symbolic_kwargs = {
                'input_scale': input_quant_symbolic_kwargs['output_scale'],
                'input_zero_point': input_quant_symbolic_kwargs['output_zero_point'],
                'input_axis': input_quant_symbolic_kwargs['input_axis']}
        else:
            input_redequant_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'input_quant_symbolic_kwargs': input_quant_symbolic_kwargs,
            'input_clip_symbolic_kwargs': input_clip_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'input_redequant_symbolic_kwargs': input_redequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs,
            'output_clip_symbolic_kwargs': output_clip_symbolic_kwargs,
            'output_dequant_symbolic_kwargs': output_dequant_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        input_quant_symbolic_kwargs = self.symbolic_kwargs['input_quant_symbolic_kwargs']
        input_clip_symbolic_kwargs = self.symbolic_kwargs['input_clip_symbolic_kwargs']
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        input_redequant_symbolic_kwargs = self.symbolic_kwargs['input_redequant_symbolic_kwargs']
        if input_dequant_symbolic_kwargs is not None:
            inp = DequantizeLinearFn.apply(inp, *input_dequant_symbolic_kwargs.values())
        if input_quant_symbolic_kwargs is not None:
            inp = QuantizeLinearFn.apply(inp, *input_quant_symbolic_kwargs.values())
            inp = DequantizeLinearFn.apply(inp, *input_redequant_symbolic_kwargs.values())
            if input_clip_symbolic_kwargs is not None:
                inp = IntClipFn.apply(inp, *input_clip_symbolic_kwargs.values())
        return inp

    def output_symbolic_execution(self, out: Tensor):
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        output_dequant_symbolic_kwargs = self.symbolic_kwargs['output_dequant_symbolic_kwargs']
        output_clip_symbolic_kwargs = self.symbolic_kwargs['output_clip_symbolic_kwargs']
        if output_quant_symbolic_kwargs is not None:
            out = QuantizeLinearFn.apply(out, *output_quant_symbolic_kwargs.values())
            if output_clip_symbolic_kwargs is not None:
                out = IntClipFn.apply(out, *output_clip_symbolic_kwargs.values())
        if output_dequant_symbolic_kwargs is not None:
            out = DequantizeLinearFn.apply(out, *output_dequant_symbolic_kwargs.values())
        return out


class StdQOpONNXQuantReLUHandler(StdQOpONNXQuantNLALHandler):
    handled_layer = QuantReLU
    input_quant_supported = True

    def op_symbolic_execution(self, inp: Tensor):
        return torch.relu(inp)


class StdQOpONNXQuantTanhHandler(StdQOpONNXQuantNLALHandler):
    handled_layer = QuantTanh
    input_quant_supported = True

    def op_symbolic_execution(self, inp: Tensor):
        return torch.tanh(inp)


class StdQOpONNXQuantSigmoidHandler(StdQOpONNXQuantNLALHandler):
    handled_layer = QuantSigmoid
    input_quant_supported = True

    def op_symbolic_execution(self, inp: Tensor):
        return torch.sigmoid(inp)


class StdQOpONNXQuantIdentityHandler(StdQOpONNXQuantNLALHandler):
    handled_layer = QuantIdentity
    input_quant_supported = False

    def op_symbolic_execution(self, inp: Tensor):
        return inp


class StdQOpONNXQuantHardTanhHandler(StdQOpONNXQuantIdentityHandler):
    handled_layer = QuantHardTanh
