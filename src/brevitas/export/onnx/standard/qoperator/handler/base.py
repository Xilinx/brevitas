from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor

from brevitas.export.common.handler.base import BitWidthHandlerMixin
from brevitas.export.common.handler.base import ClipMixin
from brevitas.export.common.handler.base import QuantAxisMixin
from brevitas.export.common.handler.base import ScaleHandlerMixin
from brevitas.export.common.handler.base import ZeroPointHandlerMixin
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.standard.function import DequantizeLinearFn
from brevitas.export.onnx.standard.function import IntClipFn
from brevitas.export.onnx.standard.function import QuantizeLinearFn
from brevitas.nn.quant_layer import QuantNonLinearActLayer


class StdQOpONNXQuantLayerHandler(ONNXBaseHandler,
                                  QuantAxisMixin,
                                  ScaleHandlerMixin,
                                  ClipMixin,
                                  BitWidthHandlerMixin,
                                  ZeroPointHandlerMixin,
                                  ABC):

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
    def torch_8b_dtype(cls, is_signed):
        if is_signed:
            return torch.int8
        else:
            return torch.uint8

    @classmethod
    def quant_output_shape(cls, module):
        cached_out = module.output_quant._cached_act
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required to export")
        return cached_out.shape

    @classmethod
    def output_quant_symbolic_kwargs(cls, module):
        quant_proxy = module.act_quant if isinstance(
            module, QuantNonLinearActLayer) else module.output_quant
        if quant_proxy.is_quant_enabled:
            return {
                'output_scale': quant_proxy.scale(),
                'output_zero_point': cls.quant_output_zero_point(module),
                'output_dtype': cls.torch_8b_dtype(quant_proxy.is_signed),
                'output_axis': cls.quant_axis(quant_proxy.scale())}
        else:
            return None

    @classmethod
    def output_clip_symbolic_kwargs(cls, module):
        quant_proxy = module.act_quant if isinstance(
            module, QuantNonLinearActLayer) else module.output_quant
        if quant_proxy.is_quant_enabled:
            narrow = quant_proxy.is_narrow_range
            signed = quant_proxy.is_signed
            bit_width = quant_proxy.bit_width()
            return cls.int_clip_symbolic_kwargs(narrow, signed, bit_width)
        else:
            return None

    @classmethod
    def input_clip_symbolic_kwargs(cls, module):
        if module.input_quant.is_quant_enabled:
            narrow = module.input_quant.is_narrow_range
            signed = module.input_quant.is_signed
            bit_width = module.input_quant.bit_width()
            return cls.int_clip_symbolic_kwargs(narrow, signed, bit_width)
        else:
            return None

    @classmethod
    def output_dequant_symbolic_kwargs(cls, module):
        return {
            'input_scale': module.output_quant.scale(),
            'input_zero_point': cls.quant_output_zero_point(module),
            'input_axis': cls.quant_axis(module.output_quant.scale())}

    @classmethod
    def input_quant_symbolic_kwargs(cls, module):
        if module.input_quant.is_quant_enabled:
            return {
                'output_scale': module.input_quant.scale(),
                'output_zero_point': cls.quant_input_zero_point(module),
                'output_dtype': cls.torch_8b_dtype(module.input_quant.is_signed),
                'output_axis': cls.quant_axis(module.input_quant.scale())}
        else:
            return None

    @classmethod
    def input_dequant_symbolic_kwargs(cls, module):
        if module.input_quant._cached_act is not None and not module.input_quant.is_quant_enabled:
            return cls.dequant_symbolic_kwargs_from_cached_io(module.input_quant._cached_act)
        else:
            return None

    @classmethod
    def dequant_symbolic_kwargs_from_cached_io(cls, cached_io):
        cls.validate_8b_bit_width(cached_io.bit_width, le_then=True)
        return {
            'input_scale':
                cached_io.scale,
            'input_zero_point':
                cls.zero_point_with_dtype(
                    cached_io.signed, cached_io.bit_width, cached_io.zero_point),
            'input_axis':
                cls.quant_axis(cached_io.scale)}

    @classmethod
    def quant_symbolic_kwargs_from_cached_io(cls, cached_io):
        cls.validate_8b_bit_width(cached_io.bit_width, le_then=True)
        q_kwargs = {
            'output_scale':
                cached_io.scale,
            'output_zero_point':
                cls.zero_point_with_dtype(
                    cached_io.signed, cached_io.bit_width, cached_io.zero_point),
            'output_dtype':
                cls.torch_8b_dtype(cached_io.signed),
            'output_axis':
                cls.quant_axis(cached_io.scale)}
        # TODO support narrow caching
        # Assume narrow is False since we are preventing it everywhere else
        int_clip_kwargs = cls.clip_symbolic_kwargs(False, cached_io.signed, cached_io.bit_width)
        return q_kwargs, int_clip_kwargs

    def symbolic_execution(self, inp: Tensor):
        inp = self.input_symbolic_execution(inp)
        out = self.op_symbolic_execution(inp)
        ret = self.output_symbolic_execution(out)
        return ret


class StdQOpONNXQuantWrapperHandler(StdQOpONNXQuantLayerHandler, ABC):

    @classmethod
    def validate(cls, module):
        cls.validate_8b_bit_width(module.quant_input_bit_width(), le_then=True)
        cls.validate_8b_bit_width(module.quant_output_bit_width(), le_then=True)

    def prepare_for_export(self, module):
        self.validate(module)
        op_symbolic_kwargs = self.op_symbolic_kwargs(module)
        input_dequant_symbolic_kwargs = self.input_dequant_symbolic_kwargs(module)
        if module.return_quant_tensor:
            output_quant_symbolic_kwargs = self.output_quant_symbolic_kwargs(module)
            output_clip_symbolic_kwargs = self.output_clip_symbolic_kwargs(module)
        else:
            output_quant_symbolic_kwargs = None
            output_clip_symbolic_kwargs = None

        self.symbolic_kwargs = {
            'op_symbolic_kwargs': op_symbolic_kwargs,
            'input_dequant_symbolic_kwargs': input_dequant_symbolic_kwargs,
            'output_quant_symbolic_kwargs': output_quant_symbolic_kwargs,
            'output_clip_symbolic_kwargs': output_clip_symbolic_kwargs}

    def input_symbolic_execution(self, inp: Tensor):
        input_dequant_symbolic_kwargs = self.symbolic_kwargs['input_dequant_symbolic_kwargs']
        inp = DequantizeLinearFn.apply(inp, *input_dequant_symbolic_kwargs.values())
        return inp

    def output_symbolic_execution(self, out: Tensor):
        output_quant_symbolic_kwargs = self.symbolic_kwargs['output_quant_symbolic_kwargs']
        output_clip_symbolic_kwargs = self.symbolic_kwargs['output_clip_symbolic_kwargs']
        if output_quant_symbolic_kwargs is not None:
            out = QuantizeLinearFn.apply(out, *output_quant_symbolic_kwargs.values())
            if output_clip_symbolic_kwargs is not None:
                out = IntClipFn.apply(out, *output_clip_symbolic_kwargs.values())
        return out
