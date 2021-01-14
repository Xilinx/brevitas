import torch
from torch import Tensor

from ..base import BaseHandler


class Validate8BitHandler(BaseHandler):

    @classmethod
    def validate_8b_bit_width(cls, bit_width: Tensor):
        if bit_width is None:
            raise RuntimeError("Bit width cannot be None")
        bit_width = int(bit_width.item())
        if bit_width != 8:
            raise RuntimeError("Only 8b bit width supported")
        return bit_width


class TypedZeroPointHandler(BaseHandler):

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
