from abc import ABC
import math

import torch
from torch.nn import Module
from torch import Tensor


__all__ = [
    'BaseHandler',
    'BitWidthHandlerMixin',
    'ZeroPointHandlerMixin'
]


class BaseHandler(Module, ABC):

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        pass

    def reset(self):
        pass


class BitWidthHandlerMixin(object):

    @classmethod
    def validate_bit_width(cls, bit_width: Tensor, reference: int, le_then=False):
        if bit_width is None:
            raise RuntimeError("Bit width cannot be None")
        bit_width = int(bit_width.item())
        if bit_width > reference:
            raise RuntimeError(f"Bit width {bit_width} is not supported.")
        elif bit_width < reference and not le_then:
            raise RuntimeError(f"Bit width {bit_width} is not supported, should be {reference}b.")
        return bit_width

    @classmethod
    def validate_8b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 8, le_then)

    @classmethod
    def validate_16b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 16, le_then)

    @classmethod
    def validate_32b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 32, le_then)


class ScaleHandlerMixin(object):

    @classmethod
    def validate_scalar_scale(cls, scale: Tensor):
        if scale is None:
            raise RuntimeError("Scale cannot be None.")
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError("Only per-tensor scaling is supported.")
        return scale.item()

    @classmethod
    def validate_scalar_int_exponent(cls, scale: Tensor):
        cls.validate_scalar_scale(scale)
        exponent = math.log2(scale)
        if not exponent.is_integer():
            raise RuntimeError("Only power-of-two scale factors are supported.")
        exponent = int(exponent)
        return exponent

    @classmethod
    def validate_neg_scalar_int_exponent(cls, scale: Tensor):
        return - cls.validate_scalar_int_exponent(scale)


class ZeroPointHandlerMixin(object):

    @classmethod
    def zero_point_with_dtype(cls, signed, zero_point):
        if not signed:
            if (zero_point < 0).any():
                raise RuntimeError("Zero points have to be positive under unsigned quantization")
            return zero_point.type(torch.uint8)
        else:
            return zero_point.type(torch.int8)

    @classmethod
    def quant_input_zero_point(cls, module):
        signed = module.is_quant_input_signed
        zero_point = module.quant_input_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)

    @classmethod
    def quant_weight_zero_point(cls, module):
        signed = module.is_quant_weight_signed
        zero_point = module.quant_weight_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)

    @classmethod
    def quant_output_zero_point(cls, module):
        signed = module.is_quant_output_signed
        zero_point = module.quant_output_zero_point()
        return cls.zero_point_with_dtype(signed, zero_point)
